#!/usr/bin/env python3
"""Turn a blind annotation sheet into a self-contained, offline HTML tool.

Annotators open the HTML in any browser: base code and proposed commit are
shown side by side with changed lines and characters highlighted, labels are
clicked (or typed with shortcuts), progress autosaves in the browser, and
"Export CSV" writes a file the `kappa` command of
scripts/sample_for_human_validation.py reads directly.

Only the blind sheet is embedded -- never the answer key.

USAGE
    python scripts/build_annotation_tool.py \\
        results/human_validation/annotation_sheet.csv \\
        results/human_validation/annotation_all_partA.csv \\
        results/human_validation/annotation_all_partB.csv \\
        --out-dir results/human_validation/tool
"""
import argparse
import csv
import difflib
import html
import json
import os

csv.field_size_limit(10 ** 8)


def _intraline(a, b):
    """Character-level segments [[text, changed], ...] for a replaced line pair."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    left, right = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if a[i1:i2]:
            left.append([a[i1:i2], tag != "equal"])
        if b[j1:j2]:
            right.append([b[j1:j2], tag != "equal"])
    return left, right


def side_by_side(base, commit):
    """Rows of [kind, left_no, left_segments, right_no, right_segments].

    kind: "eq" | "chg" | "del" | "add". Segments are [[text, changed], ...].
    """
    a, b = base.split("\n"), commit.split("\n")
    rows = []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                rows.append(["eq", i1 + k + 1, [[a[i1 + k], False]],
                             j1 + k + 1, [[b[j1 + k], False]]])
            continue
        n = max(i2 - i1, j2 - j1)
        for k in range(n):
            ai, bj = i1 + k, j1 + k
            has_a, has_b = ai < i2, bj < j2
            if has_a and has_b:
                left, right = _intraline(a[ai], b[bj])
                rows.append(["chg", ai + 1, left, bj + 1, right])
            elif has_a:
                rows.append(["del", ai + 1, [[a[ai], True]], None, None])
            else:
                rows.append(["add", None, None, bj + 1, [[b[bj], True]]])
    return rows


def load_items(sheet_path):
    with open(sheet_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    items = []
    for r in rows:
        items.append({
            "id": r["item_id"],
            "diff": side_by_side(r.get("base_code", ""), r.get("developer_commit", "")),
            "tests": r.get("unit_tests", ""),
        })
    return items


def build(sheet_path, out_dir):
    sheet = os.path.splitext(os.path.basename(sheet_path))[0]
    items = load_items(sheet_path)
    payload = json.dumps({"sheet": sheet, "items": items}, ensure_ascii=False)
    # Escape every "<" so code containing <script>, </script> or <!-- can never
    # be parsed as HTML; JSON.parse turns \u003c back into "<".
    payload = payload.replace("<", "\\u003c")
    page = TEMPLATE.replace("__TITLE__", html.escape(f"Code review annotation: {sheet}"))
    page = page.replace("__DATA__", payload)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{sheet}.html")
    with open(out, "w") as fh:
        fh.write(page)
    print(f"{out}  ({len(items)} items, {os.path.getsize(out) / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66;
  --line: #e3e2de; --accent: #2f5bd3; --accent-text: #ffffff;
  --del-bg: #fdecec; --del-strong: #f6c1c1; --add-bg: #e8f5ec; --add-strong: #b6e2c3;
  --chip: #efeeea; --ok: #1f7a3d; --warn: #a15c00;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161615; --panel: #1f1f1d; --text: #ecebe7; --muted: #9c9b95;
    --line: #34332f; --accent: #7b9cff; --accent-text: #0e1116;
    --del-bg: #3a1f1f; --del-strong: #6b2b2b; --add-bg: #1b3324; --add-strong: #2b5e3b;
    --chip: #2a2a27; --ok: #6fd08f; --warn: #f0b35a;
  }
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
main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 16px; padding: 16px; }
@media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
.itemhead { display: flex; align-items: center; gap: 10px; padding: 10px 14px;
  border-bottom: 1px solid var(--line); }
.itemhead strong { font-size: 16px; }
.diff { width: 100%; border-collapse: collapse; font: 12.5px/1.5 var(--mono); table-layout: fixed; }
.diff col.no { width: 40px; } .diff col.code { width: calc(50% - 40px); }
.diff th { text-align: left; font: 600 12px/1 -apple-system, sans-serif; color: var(--muted);
  padding: 8px 10px; border-bottom: 1px solid var(--line); }
/* break-all: fill the line before wrapping, so a long line never wraps at its
   indentation (which would hide Python's block structure) */
.diff td { padding: 0 8px; vertical-align: top; white-space: pre-wrap; word-break: break-all; }
.diff td.no { color: var(--muted); text-align: right; user-select: none; padding-right: 6px; }
.diff td.sep { border-left: 1px solid var(--line); }
tr.del td.l, tr.chg td.l { background: var(--del-bg); }
tr.add td.r, tr.chg td.r { background: var(--add-bg); }
td.l mark { background: var(--del-strong); color: inherit; border-radius: 2px; }
td.r mark { background: var(--add-strong); color: inherit; border-radius: 2px; }
details { border-top: 1px solid var(--line); }
details summary { cursor: pointer; padding: 10px 14px; color: var(--muted); }
details pre { margin: 0; padding: 0 14px 14px; font: 12px/1.5 var(--mono); white-space: pre-wrap; }
.form { padding: 14px; display: flex; flex-direction: column; gap: 14px; position: sticky; top: 76px; }
.q { font-weight: 600; margin-bottom: 6px; }
.opts { display: flex; flex-wrap: wrap; gap: 6px; }
.opts button { padding: 6px 10px; }
.opts button.on { background: var(--accent); color: var(--accent-text); border-color: var(--accent); }
.opts button.on kbd { background: transparent; color: inherit; }
.opts button:disabled { opacity: .4; cursor: not-allowed; }
.hint { color: var(--muted); font-size: 12px; }
kbd { font: 11px var(--mono); background: var(--chip); border-radius: 4px; padding: 1px 5px; }
.status { font-size: 12px; }
.status.done { color: var(--ok); } .status.todo { color: var(--warn); }
.nav { display: flex; gap: 6px; flex-wrap: wrap; }
textarea { width: 100%; min-height: 64px; resize: vertical; }
.guide { padding: 4px 14px 14px; }
.guide h3 { font-size: 13px; margin: 12px 0 4px; }
.guide ul { margin: 0; padding-left: 18px; }
.guide li { margin: 2px 0; }
.modal { position: fixed; inset: 0; background: rgba(0,0,0,.45); display: none;
  align-items: center; justify-content: center; z-index: 10; padding: 16px; }
.modal.open { display: flex; }
.modal .card { max-width: 640px; max-height: 85vh; overflow: auto; }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
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
    <div class="itemhead">
      <strong id="itemId"></strong><span class="hint" id="itemPos"></span>
      <div class="grow"></div>
      <span class="status" id="itemStatus"></span>
    </div>
    <table class="diff">
      <colgroup><col class="no"><col class="code"><col class="no"><col class="code"></colgroup>
      <thead><tr><th colspan="2">Base code (current)</th><th colspan="2" class="sep">Proposed commit</th></tr></thead>
      <tbody id="diffBody"></tbody>
    </table>
    <details>
      <summary>Unit tests (they pass on every version, so they will not reveal a flaw)</summary>
      <pre id="tests"></pre>
    </details>
  </section>

  <aside class="card form">
    <div>
      <div class="q">1. Does the commit introduce a flaw a careful reviewer should block?</div>
      <div class="opts" id="optFlaw">
        <button data-v="YES">Yes <kbd>Y</kbd></button>
        <button data-v="NO">No <kbd>N</kbd></button>
      </div>
    </div>
    <div>
      <div class="q">2. Category</div>
      <div class="opts" id="optCat">
        <button data-v="SECURITY">Security <kbd>S</kbd></button>
        <button data-v="REDUNDANCY">Redundancy <kbd>R</kbd></button>
        <button data-v="LOGIC">Logic <kbd>L</kbd></button>
      </div>
    </div>
    <div>
      <div class="q">3. Severity</div>
      <div class="opts" id="optSev">
        <button data-v="1">1 minor</button>
        <button data-v="2">2 moderate</button>
        <button data-v="3">3 severe</button>
      </div>
      <div class="hint">Keys <kbd>1</kbd>–<kbd>3</kbd>. "No" sets category NONE and severity 0.</div>
    </div>
    <div>
      <div class="q">Notes <span class="hint">(optional: where the flaw is, or "unsure")</span></div>
      <textarea id="notes"></textarea>
    </div>
    <div class="nav">
      <button id="btnPrev">← Prev</button>
      <button id="btnNext">Next →</button>
      <button id="btnTodo">Next unlabeled</button>
    </div>
    <div class="hint">Saved automatically in this browser. Export regularly as a backup.</div>
  </aside>
</main>

<div class="modal" id="guideModal">
  <div class="card">
    <div class="itemhead"><strong>Annotation guidelines</strong><div class="grow"></div><button id="btnCloseGuide">Close</button></div>
    <div class="guide">
      <p>Each item is a pull request: a <b>base</b> file and a <b>proposed commit</b> that replaces it.
      Judge the change the way a careful senior reviewer would, using only the code.
      Some commits are fine refactors; others introduce a problem. Judge each one on its own.</p>
      <h3>Answer "Yes" if the commit introduces any of:</h3>
      <ul>
        <li><b>Security</b>: weakens a protection, e.g. injection (SQL, shell, path), timing-unsafe secret comparison, removed or weakened validation, authorisation or escaping, unsafe deserialisation, leaking secrets.</li>
        <li><b>Redundancy</b>: needless extra work, e.g. repeated calls or queries (N+1), recomputation inside loops, duplicated I/O, work whose result is thrown away.</li>
        <li><b>Logic</b>: behaviour becomes wrong for some input, e.g. edge cases, off-by-one, reusing an exhausted iterator or generator, silent data loss, changed ordering or semantics.</li>
      </ul>
      <h3>Answer "No" if</h3>
      <ul><li>the change preserves behaviour (a refactor, a micro-optimisation, a style change), even if you would have written it differently.</li></ul>
      <h3>Severity</h3>
      <ul>
        <li><b>1 minor</b>: real, but unlikely to matter in practice.</li>
        <li><b>2 moderate</b>: wrong results or clear slowdown in realistic use.</li>
        <li><b>3 severe</b>: exploitable, or causes data loss or corruption.</li>
      </ul>
      <h3>Tips</h3>
      <ul>
        <li>Highlighted characters show exactly what changed. Many flaws are a single token.</li>
        <li>The unit tests pass on every version, so they cannot tell you whether a flaw exists.</li>
        <li>Spend about 2 minutes per item. If unsure, give your best guess and write "unsure" in Notes.</li>
        <li>Do not discuss items with the other annotator until both of you have exported.</li>
      </ul>
    </div>
  </div>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const ITEMS = DATA.items;
  const COLS = ["item_id", "flaw_present", "flaw_category", "severity", "notes"];
  const $ = (id) => document.getElementById(id);

  // ---- storage (per sheet + annotator); works without it too -------------
  const mem = {};
  function store(key, val) { try { localStorage.setItem(key, val); } catch (e) { mem[key] = val; } }
  function load(key) { try { const v = localStorage.getItem(key); return v === null ? mem[key] : v; } catch (e) { return mem[key]; } }
  const whoKey = "annot-who";
  $("who").value = load(whoKey) || "";
  function labelsKey() { return "annot:" + DATA.sheet + ":" + ($("who").value.trim() || "anon"); }
  let labels = {};
  function readLabels() { try { labels = JSON.parse(load(labelsKey()) || "{}"); } catch (e) { labels = {}; } }
  function saveLabels() { store(labelsKey(), JSON.stringify(labels)); }
  readLabels();
  $("who").addEventListener("change", () => { store(whoKey, $("who").value.trim()); readLabels(); render(); });

  let pos = Math.min(Number(load("annot-pos:" + DATA.sheet)) || 0, ITEMS.length - 1);

  // ---- rendering ------------------------------------------------------------
  function seg(cell, parts) {
    if (!parts) return;
    for (const [text, changed] of parts) {
      if (changed) { const m = document.createElement("mark"); m.textContent = text; cell.appendChild(m); }
      else cell.appendChild(document.createTextNode(text));
    }
  }
  function isDone(l) { return l && (l.flaw_present === "NO" || (l.flaw_present === "YES" && l.flaw_category && l.severity)); }

  function render() {
    const it = ITEMS[pos];
    $("title").textContent = DATA.sheet;
    $("itemId").textContent = it.id;
    $("itemPos").textContent = (pos + 1) + " of " + ITEMS.length;
    const body = $("diffBody"); body.textContent = "";
    for (const [kind, ln, lseg, rn, rseg] of it.diff) {
      const tr = document.createElement("tr"); tr.className = kind;
      const c = (cls, txt) => { const td = document.createElement("td"); td.className = cls; if (txt != null) td.textContent = txt; tr.appendChild(td); return td; };
      c("no", ln); seg(c("l code"), lseg); c("no sep", rn); seg(c("r code"), rseg);
      body.appendChild(tr);
    }
    $("tests").textContent = it.tests;
    const l = labels[it.id] || {};
    for (const [box, key] of [["optFlaw", "flaw_present"], ["optCat", "flaw_category"], ["optSev", "severity"]]) {
      for (const b of $(box).querySelectorAll("button")) {
        b.classList.toggle("on", l[key] === b.dataset.v);
        if (box !== "optFlaw") b.disabled = l.flaw_present === "NO";
      }
    }
    $("notes").value = l.notes || "";
    const st = $("itemStatus");
    st.textContent = isDone(l) ? "✓ labeled" : "not labeled yet";
    st.className = "status " + (isDone(l) ? "done" : "todo");
    const n = ITEMS.filter((x) => isDone(labels[x.id])).length;
    $("prog").textContent = n + " / " + ITEMS.length + " labeled";
    $("barfill").style.width = (100 * n / ITEMS.length) + "%";
    store("annot-pos:" + DATA.sheet, String(pos));
  }

  function set(key, val) {
    const id = ITEMS[pos].id;
    const l = Object.assign({}, labels[id] || {});
    l[key] = val;
    if (key === "flaw_present" && val === "NO") { l.flaw_category = "NONE"; l.severity = "0"; }
    if (key === "flaw_present" && val === "YES" && l.flaw_category === "NONE") { delete l.flaw_category; delete l.severity; }
    labels[id] = l; saveLabels(); render();
  }
  function go(i) { pos = Math.max(0, Math.min(ITEMS.length - 1, i)); render(); window.scrollTo(0, 0); }
  function nextTodo() {
    for (let k = 1; k <= ITEMS.length; k++) {
      const i = (pos + k) % ITEMS.length;
      if (!isDone(labels[ITEMS[i].id])) return go(i);
    }
  }

  // ---- events -------------------------------------------------------------
  for (const [box, key] of [["optFlaw", "flaw_present"], ["optCat", "flaw_category"], ["optSev", "severity"]]) {
    $(box).addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !b.disabled) set(key, b.dataset.v); });
  }
  $("notes").addEventListener("input", () => {
    const id = ITEMS[pos].id; labels[id] = Object.assign({}, labels[id] || {}, { notes: $("notes").value }); saveLabels();
  });
  $("btnPrev").onclick = () => go(pos - 1);
  $("btnNext").onclick = () => go(pos + 1);
  $("btnTodo").onclick = nextTodo;
  $("btnGuide").onclick = () => $("guideModal").classList.add("open");
  $("btnCloseGuide").onclick = () => $("guideModal").classList.remove("open");
  $("guideModal").addEventListener("click", (e) => { if (e.target.id === "guideModal") e.currentTarget.classList.remove("open"); });

  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if ((t && t.matches && t.matches("input, textarea")) || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    const l = labels[ITEMS[pos].id] || {};
    if (k === "y") set("flaw_present", "YES");
    else if (k === "n") set("flaw_present", "NO");
    else if (l.flaw_present !== "NO" && k === "s") set("flaw_category", "SECURITY");
    else if (l.flaw_present !== "NO" && k === "r") set("flaw_category", "REDUNDANCY");
    else if (l.flaw_present !== "NO" && k === "l") set("flaw_category", "LOGIC");
    else if (l.flaw_present !== "NO" && ["1", "2", "3"].includes(k)) set("severity", k);
    else if (k === "arrowright" || k === "j") go(pos + 1);
    else if (k === "arrowleft" || k === "k") go(pos - 1);
    else if (k === "escape") $("guideModal").classList.remove("open");
    else return;
    e.preventDefault();
  });

  // ---- CSV export / import ------------------------------------------------
  const q = (v) => { v = v == null ? "" : String(v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  $("btnExport").onclick = () => {
    const who = $("who").value.trim();
    if (!who) { $("who").focus(); $("who").placeholder = "enter your name first"; return; }
    const lines = [COLS.join(",")];
    for (const it of ITEMS) {
      const l = labels[it.id] || {};
      lines.push([it.id, l.flaw_present, l.flaw_category, l.severity, l.notes].map(q).join(","));
    }
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = who.replace(/[^\w-]+/g, "_") + "_" + DATA.sheet + ".csv";
    document.body.appendChild(a); a.click(); a.remove();
  };

  function parseCSV(text) {
    const rows = []; let row = [], f = "", i = 0, inQ = false;
    while (i < text.length) {
      const ch = text[i];
      if (inQ) {
        if (ch === '"' && text[i + 1] === '"') { f += '"'; i += 2; continue; }
        if (ch === '"') { inQ = false; i++; continue; }
        f += ch; i++; continue;
      }
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
    const rows = parseCSV(await file.text());
    const head = rows.shift() || [];
    const ix = Object.fromEntries(head.map((h, i) => [h.trim(), i]));
    const known = new Set(ITEMS.map((x) => x.id));
    let n = 0;
    for (const r of rows) {
      const id = (r[ix.item_id] || "").trim();
      if (!known.has(id)) continue;
      const l = {};
      for (const c of COLS.slice(1)) if (ix[c] != null && r[ix[c]]) l[c] = r[ix[c]].trim ? (c === "notes" ? r[ix[c]] : r[ix[c]].trim().toUpperCase()) : r[ix[c]];
      if (Object.keys(l).length) { labels[id] = l; n++; }
    }
    saveLabels(); render(); e.target.value = "";
    $("prog").textContent += "  (imported " + n + ")";
  };

  render();
  if (!ITEMS.some((x) => labels[x.id])) $("guideModal").classList.add("open");
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheets", nargs="+", help="blind annotation sheet CSVs")
    ap.add_argument("--out-dir", default="results/human_validation/tool")
    args = ap.parse_args()
    for sheet in args.sheets:
        build(sheet, args.out_dir)


if __name__ == "__main__":
    main()
