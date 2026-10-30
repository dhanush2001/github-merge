#!/usr/bin/env python3
"""Blind pair review: the ~125 pairs that need human eyes, as one easy HTML tool.

Instead of judging 600 versions one at a time, an annotator sees ONE screen per
pair: Version A and Version B side by side (both start from the same base, so
the highlighted lines are exactly where they differ), and answers

    Which version has a flaw a careful reviewer should block?  A / B / Both / Neither
    Severity of the worst flaw: 1-3

A and B are assigned at random per pair and pairs are shuffled, so the tool
never reveals which version is the intended flaw or why the pair was chosen.

Pairs selected (see results/flaw_witness/witness_verdicts.csv):
  no_diff          all pairs where no witness test found a behavioural difference
  suspect_ctrl     confirmed pairs whose clean twin was rejected by >80% of reviews
  confirmed_sample random confirmed pairs, stratified by category, to measure how
                   often a "confirmed" witness verdict is right

  python scripts/build_pair_review.py build            # sheet + key + HTML
  python scripts/build_pair_review.py score --sheets a.csv b.csv
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_annotation_tool import side_by_side  # noqa: E402
from flaw_witness import load_pairs  # noqa: E402

csv.field_size_limit(10 ** 8)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HV = os.path.join(ROOT, "results", "human_validation")
VERDICTS = os.path.join(ROOT, "results", "flaw_witness", "witness_verdicts.csv")
SUSPECTS = os.path.join(HV, "suspect_scenarios.csv")
SHEET = os.path.join(HV, "pair_review_sheet.csv")
KEY = os.path.join(HV, "pair_review_key.csv")
TOOL = os.path.join(HV, "tool", "pair_review.html")
ANSWERS = ("A", "B", "BOTH", "NEITHER")


def _pid(sid):
    return re.sub(r"_(TRAP|CTRL)_", "_", sid)


def select(per_category, seed):
    verdicts = {r["pair_id"]: r for r in csv.DictReader(open(VERDICTS))}
    confirmed = {p for p, r in verdicts.items() if r["verdict"].startswith("CONFIRMED")}
    groups = {p: "no_diff" for p, r in verdicts.items() if not r["verdict"].startswith("CONFIRMED")}
    for r in csv.DictReader(open(SUSPECTS)):
        p = _pid(r["scenario_id"])
        if "_CTRL_" in r["scenario_id"] and p in confirmed:
            groups[p] = "suspect_ctrl"
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for p in sorted(confirmed - set(groups)):
        by_cat[verdicts[p]["category"]].append(p)
    for cat in sorted(by_cat):
        for p in rng.sample(by_cat[cat], min(per_category, len(by_cat[cat]))):
            groups[p] = "confirmed_sample"
    return groups, verdicts


def cmd_build(args):
    groups, verdicts = select(args.per_category, args.seed)
    pairs = {p["pair_id"]: p for p in load_pairs()}
    rng = random.Random(args.seed + 1)
    order = sorted(groups)
    rng.shuffle(order)

    items, key_rows, sheet_rows = [], [], []
    for i, pid in enumerate(order, 1):
        p = pairs[pid]
        item_id = f"P{i:03d}"
        trap_is_a = rng.random() < 0.5
        a, b = (p["trap"], p["ctrl"]) if trap_is_a else (p["ctrl"], p["trap"])
        items.append({"id": item_id, "ab": side_by_side(a, b),
                      "base_a": side_by_side(p["base"], a), "base_b": side_by_side(p["base"], b)})
        sheet_rows.append({"item_id": item_id, "base_code": p["base"], "version_a": a, "version_b": b,
                           "answer": "", "severity": "", "notes": ""})
        key_rows.append({"item_id": item_id, "pair_id": pid, "trap_id": p["trap_id"], "ctrl_id": p["ctrl_id"],
                         "category": p["category"], "group": groups[pid],
                         "flawed_version": "A" if trap_is_a else "B",
                         "witness_verdict": verdicts[pid]["verdict"]})

    for path, rows in ((SHEET, sheet_rows), (KEY, key_rows)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    payload = json.dumps({"sheet": "pair_review", "items": items}, ensure_ascii=False)
    payload = payload.replace("<", "\\u003c")  # code may contain </script>
    os.makedirs(os.path.dirname(TOOL), exist_ok=True)
    with open(TOOL, "w") as fh:
        fh.write(TEMPLATE.replace("__DATA__", payload))

    print(f"{len(items)} pairs: {dict(Counter(groups.values()))}")
    print(f"  by category: {dict(Counter(r['category'] for r in key_rows))}")
    print(f"  tool  -> {TOOL}  ({os.path.getsize(TOOL) / 1e6:.1f} MB)  <- send this")
    print(f"  sheet -> {SHEET}")
    print(f"  key   -> {KEY}  (keep private: says which version is flawed)")


def _kappa(x, y):
    n = len(x)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_score(args):
    key = {r["item_id"]: r for r in csv.DictReader(open(KEY))}
    sheets = {}
    for path in args.sheets:
        rows = {r["item_id"]: r for r in csv.DictReader(open(path))
                if r.get("answer", "").strip().upper() in ANSWERS}
        name = os.path.basename(path)
        if name in sheets:
            name = f"{name}#{len(sheets) + 1}"
        sheets[name] = rows
        print(f"{name}: {len(rows)} answered")

    def outcome(item, ans):
        """Answer relative to the intended label."""
        if ans in ("BOTH", "NEITHER"):
            return {"BOTH": "clean_twin_flawed_too", "NEITHER": "no_flaw"}[ans]
        return "agrees" if ans == key[item]["flawed_version"] else "picked_clean_twin"

    out_rows = []
    for name, rows in sheets.items():
        print(f"\n== {name}")
        tab = defaultdict(Counter)
        for item, r in rows.items():
            ans = r["answer"].strip().upper()
            o = outcome(item, ans)
            tab[key[item]["group"]][o] += 1
            out_rows.append({**key[item], "annotator": name, "answer": ans, "outcome": o,
                             "severity": r.get("severity", ""), "notes": r.get("notes", "")})
        for g in ("confirmed_sample", "suspect_ctrl", "no_diff"):
            c = tab[g]
            n = sum(c.values())
            if n:
                print(f"  {g:17s} n={n:3d}  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items()))
                      + f"   agrees {100 * c['agrees'] / n:.0f}%")

    names = list(sheets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sheets[names[i]], sheets[names[j]]
            common = sorted(set(a) & set(b))
            k = _kappa([a[x]["answer"].strip().upper() for x in common],
                       [b[x]["answer"].strip().upper() for x in common])
            print(f"\nkappa {names[i]} vs {names[j]}: {k:.3f} on {len(common)} shared pairs")

    out = os.path.join(HV, "pair_review_scored.csv")
    if out_rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nper-pair results -> {out}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pair Review</title>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66;
  --line: #e3e2de; --accent: #2f5bd3; --accent-text: #ffffff;
  --a-bg: #fdf1e3; --a-strong: #f4c98f; --b-bg: #e6eefc; --b-strong: #a9c2f2;
  --chip: #efeeea; --ok: #1f7a3d; --warn: #a15c00;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161615; --panel: #1f1f1d; --text: #ecebe7; --muted: #9c9b95;
    --line: #34332f; --accent: #7b9cff; --accent-text: #0e1116;
    --a-bg: #3a2a17; --a-strong: #6e4a1c; --b-bg: #18263d; --b-strong: #2c4a7a;
    --chip: #2a2a27; --ok: #6fd08f; --warn: #f0b35a;
  }
}
:root[data-theme="dark"] {
  --bg: #161615; --panel: #1f1f1d; --text: #ecebe7; --muted: #9c9b95;
  --line: #34332f; --accent: #7b9cff; --accent-text: #0e1116;
  --a-bg: #3a2a17; --a-strong: #6e4a1c; --b-bg: #18263d; --b-strong: #2c4a7a;
  --chip: #2a2a27; --ok: #6fd08f; --warn: #f0b35a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--panel);
  border-bottom: 1px solid var(--line); padding: 10px 16px;
  display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center; }
header h1 { font-size: 15px; margin: 0; font-weight: 600; }
.grow { flex: 1; }
input[type=text], textarea { font: inherit; color: var(--text); background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; }
button { font: inherit; cursor: pointer; border-radius: 6px; padding: 6px 12px;
  border: 1px solid var(--line); background: var(--panel); color: var(--text); }
button:hover { border-color: var(--accent); }
button.primary { background: var(--accent); color: var(--accent-text); border-color: var(--accent); }
.progress { min-width: 180px; }
.bar { height: 6px; background: var(--chip); border-radius: 3px; overflow: hidden; margin-top: 4px; }
.bar > div { height: 100%; background: var(--accent); width: 0; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; padding: 16px; }
@media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; min-width: 0; }
.itemhead { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-bottom: 1px solid var(--line); }
.itemhead strong { font-size: 16px; }
.scroll { overflow-x: auto; }
.diff { width: 100%; border-collapse: collapse; font: 12.5px/1.5 var(--mono); table-layout: fixed; }
.diff col.no { width: 40px; } .diff col.code { width: calc(50% - 40px); }
.diff th { text-align: left; font: 600 12px/1 -apple-system, sans-serif; color: var(--muted);
  padding: 8px 10px; border-bottom: 1px solid var(--line); }
.diff td { padding: 0 8px; vertical-align: top; white-space: pre-wrap; word-break: break-all; }
.diff td.no { color: var(--muted); text-align: right; user-select: none; padding-right: 6px; }
.diff td.sep { border-left: 1px solid var(--line); }
.ab tr.del td.l, .ab tr.chg td.l { background: var(--a-bg); }
.ab tr.add td.r, .ab tr.chg td.r { background: var(--b-bg); }
.ab td.l mark { background: var(--a-strong); color: inherit; border-radius: 2px; }
.ab td.r mark { background: var(--b-strong); color: inherit; border-radius: 2px; }
.bv tr.del td.l, .bv tr.chg td.l { background: var(--chip); }
.bv tr.add td.r, .bv tr.chg td.r { background: var(--chip); }
.bv mark { background: var(--line); color: inherit; border-radius: 2px; }
.same { padding: 10px 14px; color: var(--warn); }
details { border-top: 1px solid var(--line); }
details summary { cursor: pointer; padding: 10px 14px; color: var(--muted); }
.form { padding: 14px; display: flex; flex-direction: column; gap: 14px; position: sticky; top: 76px; align-self: start; }
.q { font-weight: 600; margin-bottom: 6px; }
.opts { display: flex; flex-wrap: wrap; gap: 6px; }
.opts button.on { background: var(--accent); color: var(--accent-text); border-color: var(--accent); }
.opts button.on kbd { background: transparent; color: inherit; }
.opts button:disabled { opacity: .4; cursor: not-allowed; }
.hint { color: var(--muted); font-size: 12px; }
kbd { font: 11px var(--mono); background: var(--chip); border-radius: 4px; padding: 1px 5px; }
.status { font-size: 12px; } .status.done { color: var(--ok); } .status.todo { color: var(--warn); }
.nav { display: flex; gap: 6px; flex-wrap: wrap; }
textarea { width: 100%; min-height: 64px; resize: vertical; }
.guide { padding: 4px 16px 16px; } .guide h3 { font-size: 13px; margin: 14px 0 4px; }
.guide ul { margin: 0; padding-left: 18px; } .guide li { margin: 3px 0; }
.modal { position: fixed; inset: 0; background: rgba(0,0,0,.45); display: none;
  align-items: center; justify-content: center; z-index: 10; padding: 16px; }
.modal.open { display: flex; }
.modal .card { max-width: 640px; max-height: 85vh; overflow: auto; }
</style>
</head>
<body>
<header>
  <h1>Pair review</h1>
  <label>Annotator <input type="text" id="who" placeholder="your name" size="12"></label>
  <div class="progress"><span id="prog"></span><div class="bar"><div id="barfill"></div></div></div>
  <div class="grow"></div>
  <button id="btnGuide">Guidelines</button>
  <button id="btnImport">Import CSV</button>
  <input type="file" id="fileImport" accept=".csv" hidden>
  <button class="primary" id="btnExport">Export CSV</button>
</header>
<main>
  <section class="card">
    <div class="itemhead"><strong id="itemId"></strong><span class="hint" id="itemPos"></span>
      <div class="grow"></div><span class="status" id="itemStatus"></span></div>
    <div class="scroll"><table class="diff ab">
      <colgroup><col class="no"><col class="code"><col class="no"><col class="code"></colgroup>
      <thead><tr><th colspan="2">Version A</th><th colspan="2" class="sep">Version B</th></tr></thead>
      <tbody id="abBody"></tbody>
    </table></div>
    <div class="same" id="same" hidden>A and B are identical apart from whitespace.</div>
    <details><summary>What Version A changed from the base code</summary>
      <div class="scroll"><table class="diff bv"><colgroup><col class="no"><col class="code"><col class="no"><col class="code"></colgroup>
      <thead><tr><th colspan="2">Base</th><th colspan="2" class="sep">Version A</th></tr></thead><tbody id="baBody"></tbody></table></div></details>
    <details><summary>What Version B changed from the base code</summary>
      <div class="scroll"><table class="diff bv"><colgroup><col class="no"><col class="code"><col class="no"><col class="code"></colgroup>
      <thead><tr><th colspan="2">Base</th><th colspan="2" class="sep">Version B</th></tr></thead><tbody id="bbBody"></tbody></table></div></details>
  </section>
  <aside class="card form">
    <div>
      <div class="q">1. Which version has a flaw a careful reviewer should block?</div>
      <div class="opts" id="optAns">
        <button data-v="A">A <kbd>A</kbd></button>
        <button data-v="B">B <kbd>B</kbd></button>
        <button data-v="BOTH">Both <kbd>T</kbd></button>
        <button data-v="NEITHER">Neither <kbd>N</kbd></button>
      </div>
    </div>
    <div>
      <div class="q">2. Severity of the worst flaw</div>
      <div class="opts" id="optSev">
        <button data-v="1">1 minor</button><button data-v="2">2 moderate</button><button data-v="3">3 severe</button>
      </div>
      <div class="hint">Keys <kbd>1</kbd>–<kbd>3</kbd>. Not needed for "Neither".</div>
    </div>
    <div>
      <div class="q">Notes <span class="hint">(optional: what the flaw is, or "unsure")</span></div>
      <textarea id="notes"></textarea>
    </div>
    <div class="nav">
      <button id="btnPrev">← Prev</button><button id="btnNext">Next →</button><button id="btnTodo">Next unanswered</button>
    </div>
    <div class="hint">Saved automatically in this browser. Export when done (and now and then as a backup).</div>
  </aside>
</main>
<div class="modal" id="guideModal"><div class="card">
  <div class="itemhead"><strong>How to review</strong><div class="grow"></div><button id="btnCloseGuide">Close</button></div>
  <div class="guide">
    <p>Each screen is one pull request written two ways, <b>Version A</b> and <b>Version B</b>. Both start from the same base code,
    so the highlighted lines are exactly where they differ. Look at those lines and decide which version, if any, introduces a problem.</p>
    <h3>Pick a version if it introduces any of:</h3>
    <ul>
      <li><b>Security</b>: injection (SQL, shell, path), timing-unsafe secret comparison, weakened validation or escaping, unsafe deserialisation, leaked secrets.</li>
      <li><b>Redundancy</b>: needless extra work, e.g. repeated calls or queries, recomputation inside loops, needless copies of large data.</li>
      <li><b>Logic</b>: wrong behaviour for some realistic input, e.g. edge cases, exhausted iterators, silent data loss, shared mutable defaults.</li>
    </ul>
    <h3>Answers</h3>
    <ul>
      <li><b>A</b> or <b>B</b>: that version is flawed and the other is fine.</li>
      <li><b>Both</b>: both versions have a flaw.</li>
      <li><b>Neither</b>: both are fine. Style differences or harmless refactors are not flaws, even if you prefer one.</li>
    </ul>
    <h3>Severity</h3>
    <ul><li><b>1</b> real but unlikely to matter, <b>2</b> wrong results or clear slowdown in realistic use, <b>3</b> exploitable or data loss.</li></ul>
    <h3>Tips</h3>
    <ul>
      <li>Expect a mix: some pairs have a clear flaw, some have none, some have a subtle one. A and B are random.</li>
      <li>About 1–2 minutes per pair. If unsure, give your best guess and write "unsure" in Notes.</li>
      <li>Keyboard: <kbd>A</kbd> <kbd>B</kbd> <kbd>T</kbd> <kbd>N</kbd>, <kbd>1</kbd>–<kbd>3</kbd>, arrows or <kbd>J</kbd>/<kbd>K</kbd> to move.</li>
      <li>Don't discuss pairs with the other annotator until both of you have exported.</li>
    </ul>
  </div>
</div></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const ITEMS = DATA.items;
  const COLS = ["item_id", "answer", "severity", "notes"];
  const $ = (id) => document.getElementById(id);
  const mem = {};
  function store(k, v) { try { localStorage.setItem(k, v); } catch (e) { mem[k] = v; } }
  function load(k) { try { const v = localStorage.getItem(k); return v === null ? mem[k] : v; } catch (e) { return mem[k]; } }
  $("who").value = load("pair-who") || "";
  const labelsKey = () => "pair:" + DATA.sheet + ":" + ($("who").value.trim() || "anon");
  let labels = {};
  function readLabels() { try { labels = JSON.parse(load(labelsKey()) || "{}"); } catch (e) { labels = {}; } }
  const saveLabels = () => store(labelsKey(), JSON.stringify(labels));
  readLabels();
  $("who").addEventListener("change", () => { store("pair-who", $("who").value.trim()); readLabels(); render(); });
  let pos = Math.min(Number(load("pair-pos:" + DATA.sheet)) || 0, ITEMS.length - 1);

  function seg(cell, parts) {
    if (!parts) return;
    for (const [text, changed] of parts) {
      if (changed) { const m = document.createElement("mark"); m.textContent = text; cell.appendChild(m); }
      else cell.appendChild(document.createTextNode(text));
    }
  }
  function fill(body, rows) {
    body.textContent = "";
    for (const [kind, ln, lseg, rn, rseg] of rows) {
      const tr = document.createElement("tr"); tr.className = kind;
      const c = (cls, txt) => { const td = document.createElement("td"); td.className = cls; if (txt != null) td.textContent = txt; tr.appendChild(td); return td; };
      c("no", ln); seg(c("l code"), lseg); c("no sep", rn); seg(c("r code"), rseg);
      body.appendChild(tr);
    }
  }
  const isDone = (l) => l && (l.answer === "NEITHER" || (l.answer && l.severity));

  function render() {
    const it = ITEMS[pos];
    $("itemId").textContent = it.id;
    $("itemPos").textContent = (pos + 1) + " of " + ITEMS.length;
    fill($("abBody"), it.ab); fill($("baBody"), it.base_a); fill($("bbBody"), it.base_b);
    $("same").hidden = it.ab.some((r) => r[0] !== "eq");
    const l = labels[it.id] || {};
    for (const b of $("optAns").querySelectorAll("button")) b.classList.toggle("on", l.answer === b.dataset.v);
    for (const b of $("optSev").querySelectorAll("button")) { b.classList.toggle("on", l.severity === b.dataset.v); b.disabled = l.answer === "NEITHER"; }
    $("notes").value = l.notes || "";
    const st = $("itemStatus"); st.textContent = isDone(l) ? "✓ answered" : "not answered yet";
    st.className = "status " + (isDone(l) ? "done" : "todo");
    const n = ITEMS.filter((x) => isDone(labels[x.id])).length;
    $("prog").textContent = n + " / " + ITEMS.length + " answered";
    $("barfill").style.width = (100 * n / ITEMS.length) + "%";
    store("pair-pos:" + DATA.sheet, String(pos));
  }
  function set(key, val) {
    const id = ITEMS[pos].id;
    const l = Object.assign({}, labels[id] || {}); l[key] = val;
    if (key === "answer" && val === "NEITHER") l.severity = "0";
    if (key === "answer" && val !== "NEITHER" && l.severity === "0") delete l.severity;
    labels[id] = l; saveLabels(); render();
  }
  function go(i) { pos = Math.max(0, Math.min(ITEMS.length - 1, i)); render(); window.scrollTo(0, 0); }
  function nextTodo() { for (let k = 1; k <= ITEMS.length; k++) { const i = (pos + k) % ITEMS.length; if (!isDone(labels[ITEMS[i].id])) return go(i); } }

  $("optAns").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) set("answer", b.dataset.v); });
  $("optSev").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !b.disabled) set("severity", b.dataset.v); });
  $("notes").addEventListener("input", () => { const id = ITEMS[pos].id; labels[id] = Object.assign({}, labels[id] || {}, { notes: $("notes").value }); saveLabels(); });
  $("btnPrev").onclick = () => go(pos - 1); $("btnNext").onclick = () => go(pos + 1); $("btnTodo").onclick = nextTodo;
  $("btnGuide").onclick = () => $("guideModal").classList.add("open");
  $("btnCloseGuide").onclick = () => $("guideModal").classList.remove("open");
  $("guideModal").addEventListener("click", (e) => { if (e.target.id === "guideModal") e.currentTarget.classList.remove("open"); });
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if ((t && t.matches && t.matches("input, textarea")) || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase(), l = labels[ITEMS[pos].id] || {};
    if (k === "a") set("answer", "A"); else if (k === "b") set("answer", "B");
    else if (k === "t") set("answer", "BOTH"); else if (k === "n") set("answer", "NEITHER");
    else if (l.answer !== "NEITHER" && ["1", "2", "3"].includes(k)) set("severity", k);
    else if (k === "arrowright" || k === "j") go(pos + 1);
    else if (k === "arrowleft" || k === "k") go(pos - 1);
    else if (k === "escape") $("guideModal").classList.remove("open");
    else return;
    e.preventDefault();
  });
  const q = (v) => { v = v == null ? "" : String(v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  $("btnExport").onclick = () => {
    const who = $("who").value.trim();
    if (!who) { $("who").focus(); $("who").placeholder = "enter your name first"; return; }
    const lines = [COLS.join(",")];
    for (const it of ITEMS) { const l = labels[it.id] || {}; lines.push([it.id, l.answer, l.severity, l.notes].map(q).join(",")); }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
    a.download = who.replace(/[^\w-]+/g, "_") + "_pair_review.csv";
    document.body.appendChild(a); a.click(); a.remove();
  };
  function parseCSV(text) {
    const rows = []; let row = [], f = "", i = 0, inQ = false;
    while (i < text.length) {
      const ch = text[i];
      if (inQ) { if (ch === '"' && text[i + 1] === '"') { f += '"'; i += 2; continue; } if (ch === '"') { inQ = false; i++; continue; } f += ch; i++; continue; }
      if (ch === '"') { inQ = true; i++; continue; }
      if (ch === ",") { row.push(f); f = ""; i++; continue; }
      if (ch === "\n" || ch === "\r") { if (ch === "\r" && text[i + 1] === "\n") i++; row.push(f); rows.push(row); row = []; f = ""; i++; continue; }
      f += ch; i++;
    }
    if (f !== "" || row.length) { row.push(f); rows.push(row); }
    return rows;
  }
  $("btnImport").onclick = () => $("fileImport").click();
  $("fileImport").onchange = async (e) => {
    const file = e.target.files[0]; if (!file) return;
    const rows = parseCSV(await file.text()); const head = rows.shift() || [];
    const ix = Object.fromEntries(head.map((h, i) => [h.trim(), i]));
    const known = new Set(ITEMS.map((x) => x.id)); let n = 0;
    for (const r of rows) {
      const id = (r[ix.item_id] || "").trim(); if (!known.has(id)) continue;
      const l = {};
      if (ix.answer != null && r[ix.answer]) l.answer = r[ix.answer].trim().toUpperCase();
      if (ix.severity != null && r[ix.severity]) l.severity = r[ix.severity].trim();
      if (ix.notes != null && r[ix.notes]) l.notes = r[ix.notes];
      if (Object.keys(l).length) { labels[id] = l; n++; }
    }
    saveLabels(); render(); e.target.value = ""; $("prog").textContent += "  (imported " + n + ")";
  };
  render();
  if (!ITEMS.some((x) => labels[x.id])) $("guideModal").classList.add("open");
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--per-category", type=int, default=13, help="confirmed pairs sampled per category")
    b.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("score")
    s.add_argument("--sheets", nargs="+", required=True, help="exported pair_review CSVs")
    args = ap.parse_args()
    cmd_build(args) if args.cmd == "build" else cmd_score(args)


if __name__ == "__main__":
    main()
