#!/usr/bin/env python3
"""Static-analysis baseline (reviewer W7): would Bandit or Semgrep block the flawed commits?

Each of the 300 pairs has one base file and two commits of it: TRAP (flawed) and
CTRL (clean). A tool "flags" a commit when it reports a finding the base code
does not already have (rule-id counts in the commit minus those in the base),
so pre-existing warnings in the base never count for or against a commit.
Bandit is run at two thresholds (any severity, medium+), Semgrep with the
vendored p/python + p/security-audit packs at two thresholds (any, WARNING+).

Per tool and category it reports, in the units of the paper's tables:
  flags_flawed  share of TRAP commits flagged   (a reviewer would block the flaw)
  flags_clean   share of CTRL commits flagged   (false alarm on the clean twin)
  discriminates share of pairs flagged on TRAP but not CTRL
for all 300 pairs and for the witness-confirmed pairs
(results/flaw_witness/witness_verdicts.csv).

Tools live in a separate venv because Semgrep's pins clash with litellm:
  python -m venv .venv-static && .venv-static/bin/pip install bandit semgrep
  python scripts/static_baselines.py
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flaw_witness import load_pairs  # noqa: E402

RULES_DIR = os.path.join(ROOT, "scripts", "semgrep_rules")
VERDICTS = os.path.join(ROOT, "results", "flaw_witness", "witness_verdicts.csv")
CONFIRMED = {"CONFIRMED", "CONFIRMED_TIMEOUT"}
VERSIONS = ("base", "trap", "ctrl")
BANDIT_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
SEMGREP_RANK = {"INFO": 1, "WARNING": 2, "ERROR": 3}
# (name, tool, minimum severity rank)
CONFIGS = [
    ("bandit_any", "bandit", 1), ("bandit_medium+", "bandit", 2),
    ("semgrep_any", "semgrep", 1), ("semgrep_warning+", "semgrep", 2),
]


def tool_path(name: str) -> str:
    local = os.path.join(ROOT, ".venv-static", "bin", name)
    path = local if os.path.exists(local) else shutil.which(name)
    if not path:
        sys.exit(f"{name} not found: python -m venv .venv-static && .venv-static/bin/pip install bandit semgrep")
    return path


def write_files(pairs, directory):
    """One file per pair and version; returns {filename: (pair_id, version)}."""
    index = {}
    for p in pairs:
        for v in VERSIONS:
            name = f"{p['pair_id']}__{v}.py"
            with open(os.path.join(directory, name), "w") as fh:
                fh.write(p[v])
            index[name] = (p["pair_id"], v)
    return index


def run_bandit(directory):
    """[(filename, rule_id, severity_rank)], [filenames bandit could not parse]."""
    out = subprocess.run([tool_path("bandit"), "-r", directory, "-f", "json", "-q"],
                         capture_output=True, text=True)
    data = json.loads(out.stdout)
    hits = [(os.path.basename(r["filename"]), r["test_id"], BANDIT_RANK[r["issue_severity"]])
            for r in data["results"]]
    return hits, [os.path.basename(e["filename"]) for e in data.get("errors", [])]


def run_semgrep(directory):
    configs = sorted(os.path.join(RULES_DIR, f) for f in os.listdir(RULES_DIR) if f.endswith(".yml"))
    cmd = [tool_path("semgrep"), "scan", "--json", "--metrics=off", "--disable-version-check",
           "--no-git-ignore", "--quiet"]
    for c in configs:
        cmd += ["--config", c]
    out = subprocess.run(cmd + [directory], capture_output=True, text=True)
    data = json.loads(out.stdout)
    hits = [(os.path.basename(r["path"]), r["check_id"], SEMGREP_RANK.get(r["extra"]["severity"], 1))
            for r in data["results"]]
    return hits, [os.path.basename(e.get("path", "")) for e in data.get("errors", []) if e.get("path")]


def new_findings(found, pid, version, min_rank):
    """Rule ids (with counts) in `version` beyond those already in the pair's base."""
    commit = Counter(r for r, k in found.get((pid, version), []) if k >= min_rank)
    base = Counter(r for r, k in found.get((pid, "base"), []) if k >= min_rank)
    return commit - base


def evaluate(pairs, findings):
    """One row per (pair, config): what the tool flagged on each commit."""
    rows = []
    for name, tool, min_rank in CONFIGS:
        found = findings[tool]
        for p in pairs:
            trap = new_findings(found, p["pair_id"], "trap", min_rank)
            ctrl = new_findings(found, p["pair_id"], "ctrl", min_rank)
            rows.append({"pair_id": p["pair_id"], "category": p["category"], "config": name,
                         "witness": p.get("witness", ""), "flags_trap": int(bool(trap)),
                         "flags_ctrl": int(bool(ctrl)),
                         "trap_rules": ";".join(sorted(trap.elements())),
                         "ctrl_rules": ";".join(sorted(ctrl.elements()))})
    return rows


def summarise(rows):
    """Rates per (subset, config, category) incl. an 'All' category."""
    out = []
    subsets = {"all_pairs": lambda r: True, "witness_confirmed": lambda r: r["witness"] in CONFIRMED}
    for subset, keep in subsets.items():
        for name, _, _ in CONFIGS:
            sel = [r for r in rows if r["config"] == name and keep(r)]
            for cat in ["All"] + sorted({r["category"] for r in sel}):
                g = [r for r in sel if cat == "All" or r["category"] == cat]
                if not g:
                    continue
                n = len(g)
                out.append({"subset": subset, "config": name, "category": cat, "n_pairs": n,
                            "flags_flawed": sum(r["flags_trap"] for r in g) / n,
                            "flags_clean": sum(r["flags_ctrl"] for r in g) / n,
                            "discriminates": sum(r["flags_trap"] and not r["flags_ctrl"] for r in g) / n})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "results", "static_baselines"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = load_pairs()
    if os.path.exists(VERDICTS):
        with open(VERDICTS) as fh:
            verdict = {r["pair_id"]: r["verdict"] for r in csv.DictReader(fh)}
        for p in pairs:
            p["witness"] = verdict.get(p["pair_id"], "")

    findings, errors = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        index = write_files(pairs, tmp)
        for tool, run in (("bandit", run_bandit), ("semgrep", run_semgrep)):
            hits, errs = run(tmp)
            found = {}
            for fname, rule, rank in hits:
                found.setdefault(index[fname], []).append((rule, rank))
            findings[tool], errors[tool] = found, sorted(set(errs))
            print(f"{tool}: {len(hits)} findings, {len(errors[tool])} files not analysed")

    rows = evaluate(pairs, findings)
    summary = summarise(rows)
    write_csv(os.path.join(args.output_dir, "per_pair.csv"), rows)
    write_csv(os.path.join(args.output_dir, "summary.csv"), summary)

    lines = [f"{'subset':<18}{'config':<18}{'category':<17}{'n':>5}{'flags_flawed':>14}"
             f"{'flags_clean':>13}{'discriminates':>15}"]
    for s in summary:
        lines.append(f"{s['subset']:<18}{s['config']:<18}{s['category']:<17}{s['n_pairs']:>5}"
                     f"{s['flags_flawed']:>14.1%}{s['flags_clean']:>13.1%}{s['discriminates']:>15.1%}")
    lines.append("\nRules firing only on the flawed commit (count of pairs, all pairs):")
    for name, _, _ in CONFIGS[::2]:
        c = Counter()
        for r in rows:
            if r["config"] == name and r["flags_trap"] and not r["flags_ctrl"]:
                c.update(f"{r['category']}: {x}" for x in set(r["trap_rules"].split(";")))
        lines.append(f"  {name}:")
        lines += [f"    {k:<80}{v:>4}" for k, v in c.most_common(15)]
    for tool, errs in errors.items():
        if errs:
            lines.append(f"\n{tool} could not analyse {len(errs)} files: {', '.join(errs[:10])}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(f"\nwrote -> {args.output_dir}/")


if __name__ == "__main__":
    main()
