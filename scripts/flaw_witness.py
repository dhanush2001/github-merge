"""Executable ground truth: one flaw-witness test per TRAP/CTRL pair.

A witness must FAIL on the flawed commit and PASS on the clean twin and the
base code, so a CONFIRMED pair has its label proven by execution rather than
by opinion. The existing unit_tests cannot do this: the generator required
every version, flawed included, to pass them.

  generate  ask an LLM for a witness per pair, verify it, feed failures back
            (up to --max-attempts); resumable, saves after every pair
  verify    re-run every stored witness (no API calls)
  report    verdict counts per category + per-pair CSV

Witnesses live in data/witnesses/flaw_witnesses.json. config.py loads only
data/*.json (not subfolders), so they are never scenarios and the admin never
sees them (tests/test_witnesses.py guards this).

Models run as a cascade (--models): each later model only gets the pairs the
earlier ones could not confirm, so NO_DIFF means four models (DeepSeek, Claude,
GPT, Gemini) all failed.
The model never decides the label -- execution does.

  python scripts/flaw_witness.py generate --dry-run      # cost estimate, no API calls
  python scripts/flaw_witness.py generate --limit 6       # pilot
  python scripts/flaw_witness.py generate                 # all 300 pairs (resumable)
  python scripts/flaw_witness.py verify
  python scripts/flaw_witness.py report
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))

from pipeline.witness_runner import CONFIRMED, CONFIRMED_TIMEOUT, verify_pair  # noqa: E402

WITNESS_PATH = os.path.join(ROOT, "data", "witnesses", "flaw_witnesses.json")
REPORT_DIR = os.path.join(ROOT, "results", "flaw_witness")
DONE = {CONFIRMED, CONFIRMED_TIMEOUT}

# Via OpenRouter: (litellm name, $/M in, $/M out, reasoning effort or None).
# The cheap reasoning model goes first; the frontier models (three families)
# only see the pairs it could not confirm.
WITNESS_MODELS = {
    "deepseek-v4-flash": ("openrouter/deepseek/deepseek-v4-flash", 0.091, 0.182, "high"),
    "claude-opus-5.5":   ("openrouter/anthropic/claude-opus-5.5", 4.00, 20.00, None),
    "gpt-5.6-sol":       ("openrouter/openai/gpt-5.6-sol", 4.00, 20.00, None),
    "gemini-4-argon":    ("openrouter/google/gemini-4-argon", 2.00, 10.00, None),
    "claude-fable-5.1":  ("openrouter/anthropic/claude-fable-5.1", 10.00, 50.00, None),
}
DEFAULT_CASCADE = ["deepseek-v4-flash", "claude-opus-5.5", "gpt-5.6-sol", "gemini-4-argon"]
EST_OUTPUT_TOKENS = 4000  # per attempt, incl. reasoning

METHOD_HINTS = {
    "Security_Trap": (
        "Use a concrete attack input and assert the safe behaviour: a traversal path such as "
        "'../../etc/passwd' must stay inside the intended directory; an injection payload against "
        "an in-memory sqlite3 database must not change or leak rows; an untrusted payload must not "
        "be deserialised with pickle. For a timing-attack flaw, wrap hmac.compare_digest (or "
        "secrets.compare_digest) with a recording function and assert the code calls it. For ReDoS, "
        "time a crafted input and assert it finishes in under 1 second."),
    "Objective_Trap": (
        "PURE INPUT/OUTPUT ONLY: call the code and assert on what a caller can observe (return "
        "values, raised exceptions, mutated arguments, files or rows written). No mocks, no "
        "patched functions or modules, no call counting -- these are blocked for logic flaws. "
        "Find the concrete input or call sequence where the flawed version returns a wrong result "
        "(call twice for mutable defaults, iterate twice for exhausted generators, check closures "
        "built in a loop, use Decimal-exact money totals, raise the custom exception that must "
        "propagate). Optionally also loop over random inputs (random.seed(0)) and assert the "
        "output equals what the base logic gives, computed inside the test."),
    "Redundancy_Trap": (
        "Redundancy does not change the output, so measure its COST: wrap the expensive function, "
        "constructor, copy.deepcopy, or database call with a counting wrapper and assert the count "
        "is at most what the base code needs (e.g. one query, one connection, one computation per "
        "distinct input)."),
}

PROMPT = """You write a FLAW-WITNESS test for a code-review benchmark.

A pull request changes BASE CODE. Two versions of the commit exist:
- FLAWED commit: deliberately introduces a {category} flaw.
- CLEAN commit: a harmless refactor of the same base code.

Write ONE Python test script that:
1. PASSES on the base code and on the clean commit.
2. FAILS with an AssertionError on the flawed commit, because of the flaw itself.

Rules:
- The script runs right after the code, in the same namespace (classes/functions are already defined).
- Use only the standard library. Monkeypatching module attributes (e.g. hmac.compare_digest)
  or wrapping functions/classes to count calls is allowed.
- Do NOT read or inspect the source code or execution (no inspect, ast, dis, __code__, __file__,
  sys.settrace, frames, __defaults__, __closure__). These are blocked: test behaviour.
- Use realistic inputs the function could receive in normal use. Do not build pathological
  objects (custom __hash__/__eq__ tricks, objects that mutate themselves) to force a difference.
- Be honest: if the flawed and clean commits behave identically for every realistic input, do
  NOT invent a trick. Return "witness_test": "" and "no_behavioural_difference": true.
- If the flawed version might raise instead of returning a wrong value, catch the exception and
  turn it into a failed assert, e.g. `try: ... except Exception as e: assert False, f"raised {{e!r}}"`.
- Use a temporary directory for any files. No network. Deterministic: seed any randomness.
- Every assert needs a short message saying what the flaw is.

How to witness this category: {method_hint}
Known flaw types for this category: {subtypes}

BASE CODE:
```python
{base}
```

FLAWED COMMIT:
```python
{trap}
```

CLEAN COMMIT:
```python
{ctrl}
```

Existing happy-path tests (all versions pass these):
```python
{unit_tests}
```
{feedback}
Return strict JSON:
{{"flaw_description": "<one sentence: what the flaw is and where>",
  "method": "behavioral" | "instrumented" | "timing",
  "no_behavioural_difference": false,
  "witness_test": "<the python script>"}}
"""

FEEDBACK = """
YOUR PREVIOUS ATTEMPT did not meet the requirement (verdict {verdict}).
Previous witness:
```python
{witness}
```
Outcomes: base={base}, clean={ctrl}, flawed={trap}
Fix it. If the clean commit failed, your test is too strict or tests implementation details.
If the flawed commit passed, your input does not trigger the flaw.
"""


def pair_id(sid: str) -> str:
    return re.sub(r"_(TRAP|CTRL)_", "_", sid)


def load_pairs() -> list[dict]:
    by_id = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "dataset_b_*.json"))):
        for s in json.load(open(path)):
            by_id[s["scenario_id"]] = s
    pairs = []
    for sid, trap in by_id.items():
        if "_TRAP_" not in sid:
            continue
        ctrl = by_id[sid.replace("_TRAP_", "_CTRL_")]
        assert trap["base_code"] == ctrl["base_code"], sid
        pairs.append({"pair_id": pair_id(sid), "trap_id": sid, "ctrl_id": ctrl["scenario_id"],
                      "category": trap["category"], "base": trap["base_code"],
                      "trap": trap["developer_commit"], "ctrl": ctrl["developer_commit"],
                      "unit_tests": trap["unit_tests"]})
    return sorted(pairs, key=lambda p: p["pair_id"])


def load_witnesses() -> dict:
    if not os.path.exists(WITNESS_PATH):
        return {}
    return {w["pair_id"]: w for w in json.load(open(WITNESS_PATH))}


_save_lock = threading.Lock()


def save_witnesses(store: dict) -> None:
    os.makedirs(os.path.dirname(WITNESS_PATH), exist_ok=True)
    with _save_lock:
        tmp = WITNESS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump([store[k] for k in sorted(store)], f, indent=2)
        os.replace(tmp, WITNESS_PATH)


def _ask(model_key: str, prompt: str) -> dict:
    import litellm
    from utils import extract_json_block, normalize_message_content

    name, _, _, effort = WITNESS_MODELS[model_key]
    extra = {"extra_body": {"reasoning": {"effort": effort}}} if effort else {}
    # No temperature/seed: Claude 5.x rejects them, and witnesses are verified by
    # execution, so sampling randomness cannot change a verdict.
    resp = litellm.completion(
        model=name, messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}, timeout=300, num_retries=3,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"), **extra)
    text = normalize_message_content(resp.choices[0].message.content)
    try:
        return json.loads(text)
    except Exception:
        return extract_json_block(text) or {}


def build_prompt(pair: dict, subtypes: dict, feedback: str = "") -> str:
    return PROMPT.format(
        category=pair["category"].replace("_Trap", "").lower(),
        method_hint=METHOD_HINTS[pair["category"]],
        subtypes="; ".join(subtypes[pair["category"]]),
        base=pair["base"], trap=pair["trap"], ctrl=pair["ctrl"],
        unit_tests=pair["unit_tests"], feedback=feedback)


def _behavioural_only(category: str) -> bool:
    return category == "Objective_Trap"


def generate_one(pair: dict, models: list, max_attempts: int, subtypes: dict) -> dict:
    """Try each model in turn (up to max_attempts each, with execution feedback)."""
    feedback, attempts = "", []
    for model_key in models:
        for attempt in range(1, max_attempts + 1):
            try:
                out = _ask(model_key, build_prompt(pair, subtypes, feedback))
            except Exception as e:
                attempts.append({"model": model_key, "verdict": "API_ERROR", "error": str(e)[:300]})
                continue
            witness = str(out.get("witness_test") or "")
            check = verify_pair(pair["base"], pair["ctrl"], pair["trap"], witness,
                                behavioural_only=_behavioural_only(pair["category"]))
            attempts.append({"model": model_key, "verdict": check["verdict"], "witness_test": witness,
                             "no_behavioural_difference": bool(out.get("no_behavioural_difference")),
                             "flaw_description": out.get("flaw_description", ""),
                             "method": out.get("method", ""), "results": check["results"]})
            if check["verdict"] in DONE or not witness.strip():
                break  # confirmed, or this model says there is no behavioural difference
            r = check["results"]
            feedback = FEEDBACK.format(
                verdict=check["verdict"], witness=witness,
                **{k: f"{r[k]['outcome']} {r[k]['detail']}".strip() for k in ("base", "ctrl", "trap")})
        if attempts and attempts[-1]["verdict"] in DONE:
            break
    real = [a for a in attempts if a["verdict"] != "API_ERROR"]
    best = next((a for a in attempts if a["verdict"] in DONE), (real or attempts)[-1])
    return {"pair_id": pair["pair_id"], "trap_id": pair["trap_id"], "ctrl_id": pair["ctrl_id"],
            "category": pair["category"], "model": best["model"],
            "verdict": best["verdict"], "flaw_description": best.get("flaw_description", ""),
            "method": best.get("method", ""), "witness_test": best.get("witness_test", ""),
            "results": best.get("results", {}), "n_attempts": len(attempts),
            "attempt_verdicts": [f"{a['model']}:{a['verdict']}" for a in attempts],
            "models_tried": sorted({a["model"] for a in attempts}),
            "models_claiming_no_difference": sorted({a["model"] for a in attempts
                                                     if a.get("no_behavioural_difference")})}


def estimate(todo: list, models: list, max_attempts: int, subtypes: dict) -> None:
    """Upper-bound cost if every pair needed every attempt of every model, and a typical case."""
    in_tok = sum(len(build_prompt(p, subtypes)) // 4 + 1500 for p in todo)  # +feedback on retries
    print(f"{len(todo)} pairs, ~{in_tok // max(1, len(todo))} input tokens per attempt")
    for m in models:
        _, pin, pout, _ = WITNESS_MODELS[m]
        per_pass = in_tok * pin / 1e6 + len(todo) * EST_OUTPUT_TOKENS * pout / 1e6
        print(f"  {m:<17} ${per_pass:7.2f} per attempt over all pairs, "
              f"worst case ${per_pass * max_attempts:7.2f}")
    # Typical: 1.5 attempts on the first model, half the pairs fall through to the
    # second, and a quarter of those to each later model.
    share, total = 1.0, 0.0
    for i, m in enumerate(models):
        _, pin, pout, _ = WITNESS_MODELS[m]
        total += share * 1.5 * (in_tok * pin / 1e6 + len(todo) * EST_OUTPUT_TOKENS * pout / 1e6)
        share *= 0.5 if i == 0 else 0.25
    print(f"Typical for this cascade: ~${total:.0f}")


def cmd_generate(args):
    from generate_dataset import CATEGORIES

    unknown = [m for m in args.models if m not in WITNESS_MODELS]
    if unknown:
        sys.exit(f"Unknown model(s) {unknown}; choose from {sorted(WITNESS_MODELS)}")
    pairs = load_pairs()
    store = load_witnesses()
    if args.ids:
        wanted = {pair_id(i) for i in args.ids}
        pairs = [p for p in pairs if p["pair_id"] in wanted]
    todo = [p for p in pairs if args.redo or store.get(p["pair_id"], {}).get("verdict") not in DONE
            and (args.retry_failed or p["pair_id"] not in store)]
    if args.limit:
        todo = todo[: args.limit]
    if args.dry_run:
        estimate(todo, args.models, args.max_attempts, CATEGORIES)
        return
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set (put it in .env)")
    print(f"{len(pairs)} pairs, {len(todo)} to generate; cascade: {' -> '.join(args.models)}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(generate_one, p, args.models, args.max_attempts, CATEGORIES): p for p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            store[rec["pair_id"]] = rec
            save_witnesses(store)
            print(f"[{i}/{len(todo)}] {rec['pair_id']}: {rec['verdict']} "
                  f"({' -> '.join(rec['attempt_verdicts'])})")
    report(store)


def cmd_verify(args):
    pairs = {p["pair_id"]: p for p in load_pairs()}
    store = load_witnesses()
    changed = 0
    for pid, rec in sorted(store.items()):
        p = pairs.get(pid)
        if p is None or not rec.get("witness_test"):
            continue
        check = verify_pair(p["base"], p["ctrl"], p["trap"], rec["witness_test"], repeats=args.repeats,
                            behavioural_only=_behavioural_only(rec["category"]))
        if check["verdict"] != rec["verdict"]:
            changed += 1
            print(f"{pid}: {rec['verdict']} -> {check['verdict']}")
        rec["verdict"], rec["results"] = check["verdict"], check["results"]
    save_witnesses(store)
    print(f"Re-verified {len(store)} witnesses, {changed} verdicts changed")
    report(store)


def report(store: dict):
    os.makedirs(REPORT_DIR, exist_ok=True)
    by_cat = defaultdict(Counter)
    for rec in store.values():
        by_cat[rec["category"]][rec["verdict"]] += 1
    verdicts = sorted({v for c in by_cat.values() for v in c})
    lines = [f"{'category':<18}" + "".join(f"{v:>19}" for v in verdicts) + f"{'confirmed %':>13}"]
    for cat in sorted(by_cat):
        c = by_cat[cat]
        n = sum(c.values())
        ok = sum(c[v] for v in DONE)
        lines.append(f"{cat:<18}" + "".join(f"{c[v]:>19}" for v in verdicts) + f"{100 * ok / n:>12.1f}%")
    methods = Counter(r.get("method", "") for r in store.values() if r["verdict"] in DONE)
    models = Counter(r.get("model", "") for r in store.values() if r["verdict"] in DONE)
    lines.append(f"\nConfirmed by method: {dict(methods)}")
    lines.append(f"Confirmed by model:  {dict(models)}")
    no_diff = sum(1 for r in store.values() if r["verdict"] not in DONE and r.get("models_claiming_no_difference"))
    lines.append(f"Unconfirmed pairs where a model said 'no behavioural difference': {no_diff}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(REPORT_DIR, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(REPORT_DIR, "witness_verdicts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "trap_id", "ctrl_id", "category", "verdict", "model", "method", "n_attempts",
                    "base", "ctrl", "trap", "flaw_description"])
        for rec in sorted(store.values(), key=lambda r: r["pair_id"]):
            r = rec.get("results", {})
            w.writerow([rec["pair_id"], rec["trap_id"], rec["ctrl_id"], rec["category"], rec["verdict"],
                        rec.get("model", ""), rec.get("method", ""), rec.get("n_attempts", ""),
                        *(r.get(k, {}).get("outcome", "") for k in ("base", "ctrl", "trap")),
                        rec.get("flaw_description", "")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--models", nargs="+", default=DEFAULT_CASCADE,
                   help=f"cascade of model keys, from {sorted(WITNESS_MODELS)}")
    g.add_argument("--dry-run", action="store_true", help="print a cost estimate and exit")
    g.add_argument("--max-attempts", type=int, default=3)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--ids", nargs="*", help="scenario or pair ids to (re)generate")
    g.add_argument("--retry-failed", action="store_true", help="also redo pairs that were not confirmed")
    g.add_argument("--redo", action="store_true", help="regenerate everything selected")
    v = sub.add_parser("verify")
    v.add_argument("--repeats", type=int, default=2)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "verify":
        cmd_verify(args)
    else:
        report(load_witnesses())


if __name__ == "__main__":
    main()
