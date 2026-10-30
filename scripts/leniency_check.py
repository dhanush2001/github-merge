#!/usr/bin/env python3
"""Is persuasion making reviewers *lenient* or *fooled*?

Every flawed PR (TRAP) has a clean twin (CTRL) with the same base code, and
Experiment 1 reviewed both under the neutral (control) and persuasive (full)
description. Signal detection theory separates the two explanations:

* d' (discrimination) = z(H) - z(F): how well the reviewer tells flawed from
  clean. Persuasion lowering d' means reviewers are *fooled*.
* c (criterion) = -(z(H) + z(F)) / 2: how much evidence the reviewer needs to
  reject. Persuasion lowering c with d' intact means reviewers are *lenient*.

H = P(reject | flawed), F = P(reject | clean), with the log-linear correction
(+0.5 / +1) so rates of 0 or 1 stay finite. We also report the
difference-in-differences of approval rates: (persuasion lift on flawed) -
(persuasion lift on clean).

CIs come from a bootstrap that resamples scenario *pairs* (a flawed PR and its
clean twin move together), matching the paired design.

USAGE
    python scripts/leniency_check.py --results-dir results \\
        --out-dir results/analysis_leniency
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variance_decomposition import CELL_KEY, EXP1_MODELS, _timestamp  # noqa: E402

ID_RE = re.compile(r"^PR_B_(SEC|RED|OBJ)_\d+_(TRAP|CTRL)_")
CATEGORY = {"SEC": "Security", "RED": "Redundancy", "OBJ": "Logic"}


def load_traces(results_dir):
    """Latest-timestamp-wins dedup of every Experiment 1 trace, flawed and clean."""
    latest = {}
    pattern = os.path.join(results_dir, "conversation_logs_*.json")
    for path in sorted(glob.glob(pattern), key=_timestamp):
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for rec in payload:
            if isinstance(rec, dict):
                latest[tuple(rec.get(k) for k in CELL_KEY)] = rec

    rows = []
    for rec in latest.values():
        if {rec.get("dev_model"), rec.get("admin_model")} - EXP1_MODELS:
            continue
        m = ID_RE.match(rec.get("scenario_id", ""))
        if not m:
            continue
        cat, kind = m.groups()
        rows.append({
            # The timestamp, not the number, makes a scenario unique.
            "pair": re.sub(r"_(TRAP|CTRL)_", "_", rec["scenario_id"]),
            "flawed": kind == "TRAP",
            "category": CATEGORY[cat],
            "admin": rec["admin_model"],
            "dev": rec["dev_model"],
            "cond": rec["persuasion_mode"],
            "approved": rec.get("final_decision") == "AdminDecision.APPROVE",
        })
    return pd.DataFrame(rows)


def sdt(df):
    """d', c, and approval rates for one slice (needs both flawed and clean)."""
    f, c = df[df["flawed"]], df[~df["flawed"]]
    h = (f["approved"].eq(False).sum() + 0.5) / (len(f) + 1)   # hit: reject flawed
    fa = (c["approved"].eq(False).sum() + 0.5) / (len(c) + 1)  # false alarm: reject clean
    return {"d_prime": norm.ppf(h) - norm.ppf(fa),
            "criterion": -(norm.ppf(h) + norm.ppf(fa)) / 2,
            "approve_flawed": f["approved"].mean(),
            "approve_clean": c["approved"].mean()}


def contrast(df):
    """Persuasion (full) minus neutral (control) for every SDT quantity."""
    a = sdt(df[df["cond"] == "full"])
    b = sdt(df[df["cond"] == "control"])
    out = {f"{k}_control": b[k] for k in b}
    out.update({f"{k}_full": a[k] for k in a})
    out.update({f"delta_{k}": a[k] - b[k] for k in a})
    out["did_approval"] = out["delta_approve_flawed"] - out["delta_approve_clean"]
    return out


def bootstrap(df, n_boot, seed, keys=("delta_d_prime", "delta_criterion", "did_approval")):
    """Pair-clustered percentile CIs for the persuasion contrasts."""
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby("pair")}
    pairs = np.array(list(groups))
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        sample = pd.concat([groups[p] for p in rng.choice(pairs, len(pairs))],
                           ignore_index=True)
        res = contrast(sample)
        for k in keys:
            draws[k].append(res[k])
    return {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in draws.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/analysis_leniency")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_traces(args.results_dir)
    counts = df.groupby(["flawed", "cond"]).size()
    print(f"traces: {len(df):,}   pairs: {df['pair'].nunique()}")
    print(counts.to_string(), "\n")

    rows = [{"slice": "pooled", **contrast(df)}]
    rows += [{"slice": f"admin={a}", **contrast(g)} for a, g in df.groupby("admin")]
    rows += [{"slice": f"category={c}", **contrast(g)} for c, g in df.groupby("category")]
    table = pd.DataFrame(rows)

    print(f"bootstrapping {args.n_boot} pair-clustered resamples ...")
    ci = bootstrap(df, args.n_boot, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "leniency_sdt.csv"), index=False)

    cols = ["slice", "approve_flawed_control", "approve_flawed_full",
            "approve_clean_control", "approve_clean_full", "did_approval",
            "d_prime_control", "d_prime_full", "delta_d_prime",
            "criterion_control", "criterion_full", "delta_criterion"]
    pooled = table.iloc[0]
    lines = [
        "Experiment 1: lenient or fooled? (signal detection on paired traces)",
        "=" * 70,
        f"traces {len(df):,}  pairs {df['pair'].nunique()}  bootstrap {args.n_boot} (pair-clustered)",
        "",
        f"approval of FLAWED  control {pooled.approve_flawed_control:.3f} -> full {pooled.approve_flawed_full:.3f}"
        f"  ({pooled.delta_approve_flawed:+.3f})",
        f"approval of CLEAN   control {pooled.approve_clean_control:.3f} -> full {pooled.approve_clean_full:.3f}"
        f"  ({pooled.delta_approve_clean:+.3f})",
        f"difference-in-differences {pooled.did_approval:+.3f}  95% CI "
        f"[{ci['did_approval'][0]:+.3f}, {ci['did_approval'][1]:+.3f}]",
        "",
        f"d'  control {pooled.d_prime_control:.3f} -> full {pooled.d_prime_full:.3f}  "
        f"delta {pooled.delta_d_prime:+.3f}  95% CI [{ci['delta_d_prime'][0]:+.3f}, {ci['delta_d_prime'][1]:+.3f}]",
        f"c   control {pooled.criterion_control:.3f} -> full {pooled.criterion_full:.3f}  "
        f"delta {pooled.delta_criterion:+.3f}  95% CI [{ci['delta_criterion'][0]:+.3f}, {ci['delta_criterion'][1]:+.3f}]",
        "",
        table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"),
    ]
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
