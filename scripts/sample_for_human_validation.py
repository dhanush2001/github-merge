#!/usr/bin/env python3
"""Sample a stratified, blind subset of scenarios for human validation, and
score inter-annotator agreement once the sheets are filled in.

The goal is construct validity: show that the GPT-4o-generated flaws and clean
controls are perceptible to independent human experts, not just to the
generator's own referees.

DESIGN
------
* Every flawed commit (``*_TRAP_*``) has a clean twin (``*_CTRL_*``) with the
  same base code and unit tests. We sample *pairs*, N per category
  (SEC / RED / OBJ), and show each annotator exactly ONE version of each pair,
  half flawed and half clean within every category. Seeing both versions
  would let annotators diff them instead of judging the code.
* The annotation sheet is blind: items get neutral ids (H001, ...). The
  scenario id (which contains TRAP/CTRL) and the category (Control_Group /
  Security_Trap / ...) go only in a separate answer key.

USAGE
-----
Draw the sheet and its answer key (fixed seed):

    python scripts/sample_for_human_validation.py sample \\
        --data-dir data --per-category 40 --seed 42 \\
        --out annotation_sheet.csv --key annotation_key.csv

Or annotate EVERY scenario (two sheets, one version of each pair per sheet;
label part A before part B so the twins are never side by side):

    python scripts/sample_for_human_validation.py sample --all \\
        --out annotation_all.csv --key annotation_all_key.csv

Give each annotator a copy of annotation_sheet.csv. Keep annotation_key.csv
away from them. Columns to fill:

    flaw_present   YES | NO
    flaw_category  SECURITY | REDUNDANCY | LOGIC | NONE
    severity       0 (none) .. 3 (severe)
    notes          free text

Score agreement:

    python scripts/sample_for_human_validation.py kappa \\
        --key annotation_key.csv --sheets annotatorA.csv annotatorB.csv

Reports Cohen's kappa between annotators (flaw_present, flaw_category,
quadratic-weighted kappa for severity) and each annotator's agreement with the
pipeline's intended label, overall and per category. No third-party
dependencies.
"""
import argparse
import csv
import glob
import json
import os
import random
import re
from collections import defaultdict

ID_RE = re.compile(r"^PR_B_(SEC|RED|OBJ)_\d+_(TRAP|CTRL)_")
CATEGORY = {"SEC": "SECURITY", "RED": "REDUNDANCY", "OBJ": "LOGIC"}

# Scenarios whose commit text gives the answer away (e.g. a "# This line is
# the trap" comment). They are unfit for a blind audit.
EXCLUDE = {"PR_B_OBJ_0001_1776897575238"}

ANNOTATION_COLUMNS = ["flaw_present", "flaw_category", "severity", "notes"]
SHEET_COLUMNS = ["item_id", "base_code", "developer_commit",
                 "unit_tests"] + ANNOTATION_COLUMNS
KEY_COLUMNS = ["item_id", "scenario_id", "pair_id", "category",
               "intended_flaw_present", "intended_flaw_category"]


def _pair_id(scenario_id):
    # The number (e.g. 0001) repeats across scenarios; the trailing
    # timestamp is what makes an id unique, so keep it.
    return re.sub(r"_(TRAP|CTRL)_", "_", scenario_id)


def _load_pairs(data_dir):
    """Return {pair_id: {"TRAP": scenario, "CTRL": scenario, "cat": code}}."""
    pairs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        try:
            items = json.load(open(path))
        except Exception as exc:
            print(f"  skip {path}: {exc}")
            continue
        for s in items:
            m = ID_RE.match(s.get("scenario_id", ""))
            if not m:
                continue
            cat, kind = m.groups()
            p = pairs[_pair_id(s["scenario_id"])]
            p[kind], p["cat"] = s, cat
    return {k: v for k, v in pairs.items() if "TRAP" in v and "CTRL" in v}


def _write_items(picks, sheet_path, key_writer, prefix):
    """Write one blind sheet; append its answers to the (shared) key."""
    with open(sheet_path, "w", newline="") as sh:
        ws = csv.DictWriter(sh, fieldnames=SHEET_COLUMNS)
        ws.writeheader()
        for n, (s, pid, cat, kind) in enumerate(picks, 1):
            item = f"{prefix}{n:03d}"
            ws.writerow({"item_id": item,
                         "base_code": s.get("base_code", ""),
                         "developer_commit": s.get("developer_commit", ""),
                         "unit_tests": s.get("unit_tests", ""),
                         **{c: "" for c in ANNOTATION_COLUMNS}})
            key_writer.writerow({"item_id": item, "scenario_id": s["scenario_id"],
                                 "pair_id": pid, "category": CATEGORY[cat],
                                 "intended_flaw_present": "YES" if kind == "TRAP" else "NO",
                                 "intended_flaw_category": CATEGORY[cat] if kind == "TRAP" else "NONE"})
    counts = defaultdict(lambda: [0, 0])
    for _, _, cat, kind in picks:
        counts[CATEGORY[cat]][kind == "CTRL"] += 1
    print(f"Wrote {len(picks)} items to {sheet_path} (blind).")
    for cat in sorted(counts):
        print(f"  {cat:<11} flawed {counts[cat][0]:>3}  clean {counts[cat][1]:>3}")


def _split_kinds(pids, rng):
    """Exactly half flawed, half clean (shuffled) for a list of pairs."""
    kinds = ["TRAP"] * (len(pids) // 2) + ["CTRL"] * (len(pids) - len(pids) // 2)
    rng.shuffle(kinds)
    return kinds


def cmd_sample(args):
    pairs = _load_pairs(args.data_dir)
    rng = random.Random(args.seed)
    by_cat = defaultdict(list)
    for pid in sorted(pairs):
        # --all covers every pair; the leaking comment has since been removed
        # from the data, so nothing needs excluding there.
        if args.all or pid not in EXCLUDE:
            by_cat[pairs[pid]["cat"]].append(pid)

    with open(args.key, "w", newline="") as kh:
        wk = csv.DictWriter(kh, fieldnames=KEY_COLUMNS)
        wk.writeheader()

        if not args.all:
            print(f"Pool: {len(pairs)} pairs; excluded {sorted(EXCLUDE & set(pairs))}.")
            picks = []
            for cat in sorted(by_cat):
                pids = rng.sample(by_cat[cat], min(args.per_category, len(by_cat[cat])))
                picks += [(pairs[pid][kind], pid, cat, kind)
                          for pid, kind in zip(pids, _split_kinds(pids, rng))]
            rng.shuffle(picks)  # interleave categories so position reveals nothing
            _write_items(picks, args.out, wk, "H")
        else:
            # Every pair, split into two sheets. Each sheet holds ONE version
            # of every pair (half flawed, half clean per category); the other
            # sheet holds its twin. Label part A before part B.
            part_a, part_b = [], []
            flip = {"TRAP": "CTRL", "CTRL": "TRAP"}
            for cat in sorted(by_cat):
                pids = by_cat[cat]
                for pid, kind in zip(pids, _split_kinds(pids, rng)):
                    part_a.append((pairs[pid][kind], pid, cat, kind))
                    part_b.append((pairs[pid][flip[kind]], pid, cat, flip[kind]))
            rng.shuffle(part_a)
            rng.shuffle(part_b)
            stem, ext = os.path.splitext(args.out)
            print(f"Pool: {len(pairs)} pairs, all included.")
            _write_items(part_a, f"{stem}_partA{ext}", wk, "A")
            _write_items(part_b, f"{stem}_partB{ext}", wk, "B")
    print(f"Answer key: {args.key} (keep away from annotators).")


def _cohen_kappa(a, b, weights=None):
    """Cohen's kappa; weights='quadratic' for ordinal integer labels."""
    assert len(a) == len(b) and a, "need equal, non-empty label lists"
    n = len(a)
    labels = sorted(set(a) | set(b))
    if weights == "quadratic":
        span = (max(labels) - min(labels)) or 1
        w = lambda x, y: ((x - y) / span) ** 2
    else:
        w = lambda x, y: 0.0 if x == y else 1.0
    do = sum(w(x, y) for x, y in zip(a, b)) / n
    de = sum((a.count(x) / n) * (b.count(y) / n) * w(x, y)
             for x in labels for y in labels)
    return 1.0 if de == 0 else 1 - do / de


def _read_sheet(path):
    with open(path, newline="") as fh:
        return {r["item_id"].strip(): {k: (r.get(k) or "").strip().upper()
                                       for k in ANNOTATION_COLUMNS}
                for r in csv.DictReader(fh) if r.get("item_id")}


def cmd_kappa(args):
    with open(args.key, newline="") as fh:
        key = {r["item_id"]: r for r in csv.DictReader(fh)}
    sheets = [_read_sheet(p) for p in args.sheets]
    names = [os.path.basename(p) for p in args.sheets]
    # Only fully labeled items: NO, or YES with a category and a severity.
    done = [{i for i, r in s.items()
             if r["flaw_present"] == "NO"
             or (r["flaw_present"] == "YES" and r["flaw_category"] not in ("", "NONE")
                 and r["severity"].isdigit())}
            for s in sheets]
    common = sorted(set(key).intersection(*done))
    if not common:
        print("No overlapping, fully-labeled items across sheets.")
        return
    print(f"Scoring {len(common)} items labeled by all {len(sheets)} sheets.\n")

    for i in range(len(sheets)):
        for j in range(i + 1, len(sheets)):
            pair = f"{names[i]} vs {names[j]}"
            a = [sheets[i][k]["flaw_present"] for k in common]
            b = [sheets[j][k]["flaw_present"] for k in common]
            print(f"kappa flaw_present   {pair}: {_cohen_kappa(a, b):.3f}")
            a = [sheets[i][k]["flaw_category"] or "NONE" for k in common]
            b = [sheets[j][k]["flaw_category"] or "NONE" for k in common]
            print(f"kappa flaw_category  {pair}: {_cohen_kappa(a, b):.3f}")
            sev = [k for k in common if sheets[i][k]["severity"].isdigit()
                   and sheets[j][k]["severity"].isdigit()]
            if sev:
                a = [int(sheets[i][k]["severity"]) for k in sev]
                b = [int(sheets[j][k]["severity"]) for k in sev]
                print(f"kappa severity (qw)  {pair}: "
                      f"{_cohen_kappa(a, b, 'quadratic'):.3f}  (n={len(sev)})")

    print("\nAgreement with intended label (flaw_present):")
    for name, s in zip(names, sheets):
        hits = defaultdict(lambda: [0, 0])
        for k in common:
            ok = s[k]["flaw_present"] == key[k]["intended_flaw_present"]
            for g in ("ALL", key[k]["category"]):
                hits[g][0] += ok
                hits[g][1] += 1
        print(f"  {name}: " + "  ".join(f"{g} {h[0] / h[1]:.1%} (n={h[1]})"
                                        for g, h in sorted(hits.items())))

    print("\nItems where every annotator disagrees with the intended label "
          "(candidates to drop or repair):")
    flagged = [k for k in common
               if all(s[k]["flaw_present"] != key[k]["intended_flaw_present"]
                      for s in sheets)]
    for k in flagged:
        print(f"  {k}  {key[k]['scenario_id']}  intended "
              f"{key[k]['intended_flaw_present']}")
    if not flagged:
        print("  none")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sample", help="draw the blind annotation sheet + key")
    ps.add_argument("--data-dir", default="data")
    ps.add_argument("--per-category", type=int, default=40,
                    help="pairs sampled per category (one version shown each)")
    ps.add_argument("--all", action="store_true",
                    help="annotate every scenario: writes <out>_partA / _partB, "
                         "each with one version of every pair")
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--out", default="annotation_sheet.csv")
    ps.add_argument("--key", default="annotation_key.csv")
    ps.set_defaults(func=cmd_sample)

    pk = sub.add_parser("kappa", help="score agreement from completed sheets")
    pk.add_argument("--key", required=True, help="answer key from `sample`")
    pk.add_argument("--sheets", nargs="+", required=True,
                    help="two or more completed annotation CSVs")
    pk.set_defaults(func=cmd_kappa)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
