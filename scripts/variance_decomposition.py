#!/usr/bin/env python3
"""Crossed variance decomposition and scenario-clustered inference for Experiment 1.

Motivation
----------
Two reviewer objections target the same gap:

  * "no crossed decomposition reports reviewer, developer, scenario, and
    interaction components" -- the admin-dominance claim currently rests on
    comparing an FPR *range* across admins (37.6 pts) with the range across
    developers (7.8 pts), plus a heatmap.  A range is not a variance.

  * "the Results section does not identify the bootstrap resampling unit or
    describe clustering for pooled McNemar tests.  Trace-level resampling
    would underestimate uncertainty."  Every admin x developer pairing is
    evaluated on the *same* 300 flawed scenarios, so traces are not
    independent; resampling traces ignores that shared structure.

This script answers both from the existing Experiment 1 traces.  It fits a
crossed random-effects model to the flawed-commit approval indicator with
random intercepts for scenario, admin model, developer model, and the
admin x developer interaction, and reports each variance component as a share
of total random variance.  It then recomputes the headline persuasion delta
under trace-level and scenario-clustered bootstrap resampling so the two
interval widths can be compared directly.

Deduplication
-------------
Several scenario/model cells were evaluated more than once (notably a partial
gpt-4.1-mini control run on 2026-05-06 that was superseded by a complete rerun
on 2026-05-12).  Records are keyed on
(scenario_id, dev_model, admin_model, persuasion_mode, dataset_label) and the
run with the latest timestamp wins, which reproduces the published per-model
FPRs exactly.  Run with --verify to check that reproduction.

Usage
-----
    python scripts/variance_decomposition.py \
        --results-dir results --output-dir results/analysis_variance

    python scripts/variance_decomposition.py --verify   # replication check only
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

EXP1_MODELS = {
    "gemma-4-31b", "deepseek-v3.2", "qwen3-32b", "gpt-4.1-mini", "llama-3.3-70b",
}

# Published Experiment 1 flaw-approval rates (Table 2), used by --verify.
PUBLISHED_FPR = {
    "llama-3.3-70b": (0.182, 0.486),
    "gpt-4.1-mini":  (0.2087, 0.372),
    "qwen3-32b":     (0.2567, 0.3307),
    "deepseek-v3.2": (0.0587, 0.134),
    "gemma-4-31b":   (0.094, 0.110),
}

CELL_KEY = ("scenario_id", "dev_model", "admin_model", "persuasion_mode", "dataset_label")


def _timestamp(path):
    m = re.search(r"(\d{8}_\d{6})", os.path.basename(path))
    return m.group(1) if m else ""


def load_experiment1(results_dir):
    """Return a DataFrame of deduplicated Experiment 1 flawed-commit traces."""
    pattern = os.path.join(results_dir, "conversation_logs_*.json")
    latest = {}
    for path in sorted(glob.glob(pattern), key=_timestamp):
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for rec in payload:
            if not isinstance(rec, dict):
                continue
            latest[tuple(rec.get(k) for k in CELL_KEY)] = rec

    rows = []
    for rec in latest.values():
        if {rec.get("dev_model"), rec.get("admin_model")} - EXP1_MODELS:
            continue
        if rec.get("dataset_type") != "DatasetType.B":
            continue
        rows.append({
            "scenario_id": rec["scenario_id"],
            "admin": rec["admin_model"],
            "dev": rec["dev_model"],
            "persuasion": rec["persuasion_mode"],
            "category": rec.get("category"),
            "approved": int(rec.get("final_decision") == "AdminDecision.APPROVE"),
        })
    df = pd.DataFrame(rows)
    df["pair"] = df["admin"] + "|" + df["dev"]
    return df


def verify(df):
    """Check the deduplicated frame reproduces the published per-model FPRs."""
    print(f"{'admin':16s} {'cond':8s} {'replicated':>11s} {'published':>10s} {'diff':>8s}")
    ok = True
    for admin, (ctrl, full) in PUBLISHED_FPR.items():
        for cond, want in (("control", ctrl), ("full", full)):
            sub = df[(df.admin == admin) & (df.persuasion == cond)]
            got = sub.approved.mean()
            delta = got - want
            ok &= abs(delta) < 0.001
            print(f"{admin:16s} {cond:8s} {got:11.4f} {want:10.4f} {delta:+8.4f}")
    pooled = {c: df[df.persuasion == c].approved.mean() for c in ("control", "full")}
    print(f"\npooled control {pooled['control']:.4f} (published 0.160)")
    print(f"pooled full    {pooled['full']:.4f} (published 0.287)")
    print(f"delta          {pooled['full'] - pooled['control']:+.4f} (published +0.127)")
    print("\nREPLICATION: " + ("EXACT" if ok else "MISMATCH"))
    return ok


def variance_components(df):
    """Crossed random-effects decomposition of the flaw-approval indicator.

    Fits a linear probability model with a single grouping constant and four
    crossed variance components, so the reported variances are in (probability)^2
    units and read directly as shares of explainable random variance.
    """
    import statsmodels.formula.api as smf

    data = df.copy()
    data["grp"] = 1
    vc = {
        "scenario": "0 + C(scenario_id)",
        "admin": "0 + C(admin)",
        "dev": "0 + C(dev)",
        "admin_x_dev": "0 + C(pair)",
    }
    model = smf.mixedlm(
        "approved ~ C(persuasion) + C(category)",
        data=data, groups="grp", vc_formula=vc,
    )
    res = model.fit(reml=True, method="lbfgs", maxiter=200)

    names = list(res.model.exog_vc.names)
    comps = {}
    for name in vc:
        if name in names:
            comps[name] = float(res.vcomp[names.index(name)])
    comps["residual"] = float(res.scale)
    return res, comps


def logit_variance_components(df):
    """Logit-scale robustness check: crossed binomial GLMM (variational Bayes)."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    vcf = {
        "scenario": "0 + C(scenario_id)",
        "admin": "0 + C(admin)",
        "dev": "0 + C(dev)",
        "admin_x_dev": "0 + C(pair)",
    }
    model = BinomialBayesMixedGLM.from_formula(
        "approved ~ C(persuasion) + C(category)", vcf, df)
    res = model.fit_vb(verbose=False)
    sds = np.exp(np.asarray(res.vcp_mean))
    return {name: float(sd) ** 2 for name, sd in zip(model.vcp_names, sds)}


COMPONENTS = ("scenario", "admin", "admin_x_dev", "dev")
LOGIT_RESIDUAL = np.pi ** 2 / 3  # latent-logistic residual variance (Nakagawa & Schielzeth, 2010)


def icc_table(comps, logit):
    """Intraclass correlation of each random effect on both scales.

    ICC_c = var_c / total variance: the share of ALL variation in the approval
    outcome attributable to factor c (equivalently, the correlation between
    two traces that share only c). Linear scale: total = random + residual
    variance of the linear probability model. Logit scale: total = random +
    pi^2/3, the standard latent-scale ICC for a binary outcome.
    """
    lin_total = sum(comps.values())
    logit_total = sum(logit.get(c, 0.0) for c in COMPONENTS) + LOGIT_RESIDUAL
    rand = lin_total - comps["residual"]
    rows = []
    for name in (*COMPONENTS, "residual"):
        var = comps.get(name, np.nan)
        lv = LOGIT_RESIDUAL if name == "residual" else logit.get(name, np.nan)
        rows.append({
            "component": name,
            "variance": var,
            "sd": np.sqrt(var),
            "icc_linear": var / lin_total,
            "pct_of_random": (100.0 * var / rand) if name != "residual" else np.nan,
            "variance_logit": lv,
            "icc_logit": lv / logit_total,
        })
    return pd.DataFrame(rows)


def marginal_spread(df):
    """Model-free check: spread of marginal cell means under persuasion."""
    f = df[df.persuasion == "full"]
    out = {}
    for label, col in (("admin", "admin"), ("dev", "dev"), ("scenario", "scenario_id")):
        m = f.groupby(col).approved.mean()
        out[label] = {"var": float(m.var(ddof=1)),
                      "range": float(m.max() - m.min())}
    return out


def write_latex(vc_df, boot, path, n_traces, n_scenarios, n_pairings):
    """Emit the variance-decomposition table as a drop-in LaTeX float."""
    order = ["scenario", "admin", "admin_x_dev", "dev"]
    pretty = {"scenario": "Scenario", "admin": "Admin agent",
              "dev": "Developer agent", "admin_x_dev": "Admin $\\times$ developer"}
    idx = vc_df.set_index("component")
    rows = []
    for name in order:
        if name not in idx.index:
            continue
        r = idx.loc[name]
        rows.append(
            f"        {pretty[name]:28s} & {r['variance']:.5f} & {100 * r['icc_linear']:5.1f} & "
            f"{r['pct_of_random']:5.1f} & {r['variance_logit']:.3f} & {100 * r['icc_logit']:5.1f} \\\\")
    body = "\n".join(rows)
    tw = boot["trace_ci"][1] - boot["trace_ci"][0]
    cw = boot["cluster_ci"][1] - boot["cluster_ci"][0]
    ratio_lin = idx.loc["admin", "variance"] / idx.loc["dev", "variance"]
    ratio_logit = idx.loc["admin", "variance_logit"] / idx.loc["dev", "variance_logit"]
    top = idx.loc[list(order)].icc_logit.idxmax()
    tex = f"""% Crossed variance decomposition for Experiment 1 (flaw-approval indicator).
% Generated by scripts/variance_decomposition.py -- do not edit by hand.
\\begin{{table}}[t]
    \\centering
    \\small
    \\setlength{{\\tabcolsep}}{{1.5mm}}
    \\begin{{tabular}}{{lccccc}}
        \\toprule
         & \\multicolumn{{3}}{{c}}{{Linear probability}} & \\multicolumn{{2}}{{c}}{{Logit}} \\\\
        \\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-6}}
        Random effect & Variance & ICC (\\%) & \\% of rand. & Variance & ICC (\\%) \\\\
        \\midrule
{body}
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Crossed variance decomposition of flawed-commit approval in
    Experiment~1 ({n_traces:,} traces, {n_scenarios} scenarios, {n_pairings} pairings),
    with random intercepts for scenario, admin agent, developer
    agent, and their interaction. ICC = component variance / total variance
    (logit residual $\\pi^2/3$). Admin-agent variance is {ratio_lin:.0f}$\\times$
    (linear) and {ratio_logit:.0f}$\\times$ (logit) the developer-agent variance.
    The largest component is {pretty[top].lower()}. The headline persuasion delta is unchanged under
    scenario-clustered resampling ($+{boot['point']:.4f}$; clustered 95\\% CI
    width {cw:.4f} vs.\\ {tw:.4f} trace-level). Scenario-clustered tests of the
    persuasion effect are reported in Table~\\ref{{tab:clustered_inference}}.}}
    \\label{{tab:variance_components}}
\\end{{table}}
"""
    with open(path, "w") as fh:
        fh.write(tex)


def clustered_bootstrap(df, n_boot=5000, seed=42):
    """Pooled persuasion delta under trace-level vs scenario-clustered resampling."""
    rng = np.random.default_rng(seed)
    full = df[df.persuasion == "full"]
    ctrl = df[df.persuasion == "control"]
    point = full.approved.mean() - ctrl.approved.mean()

    # Trace-level: resample rows independently within each condition.
    fv, cv = full.approved.to_numpy(), ctrl.approved.to_numpy()
    trace = np.empty(n_boot)
    for b in range(n_boot):
        trace[b] = (rng.choice(fv, fv.size, replace=True).mean()
                    - rng.choice(cv, cv.size, replace=True).mean())

    # Scenario-clustered: resample whole scenarios, carrying all their traces.
    by_scen = {s: (g[g.persuasion == "full"].approved.to_numpy(),
                   g[g.persuasion == "control"].approved.to_numpy())
               for s, g in df.groupby("scenario_id")}
    scenarios = np.array(list(by_scen))
    clust = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(scenarios, scenarios.size, replace=True)
        fs = np.concatenate([by_scen[s][0] for s in pick])
        cs = np.concatenate([by_scen[s][1] for s in pick])
        clust[b] = fs.mean() - cs.mean()

    def ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return {
        "point": float(point),
        "trace_ci": ci(trace), "trace_se": float(trace.std(ddof=1)),
        "cluster_ci": ci(clust), "cluster_se": float(clust.std(ddof=1)),
        "n_scenarios": int(scenarios.size), "n_boot": n_boot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output-dir", default="results/analysis_variance")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true",
                    help="only check replication of published FPRs, then exit")
    args = ap.parse_args()

    df = load_experiment1(args.results_dir)
    print(f"Experiment 1 flawed-commit traces: {len(df):,}")
    print(f"  scenarios {df.scenario_id.nunique()}  admins {df.admin.nunique()} "
          f" developers {df.dev.nunique()}  pairings {df.pair.nunique()}\n")

    if args.verify:
        sys.exit(0 if verify(df) else 1)

    if not verify(df):
        print("\nWARNING: replication mismatch; downstream numbers are suspect.\n")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    res, comps = variance_components(df)
    logit = logit_variance_components(df)
    vc_df = icc_table(comps, logit)
    vc_df.to_csv(os.path.join(args.output_dir, "variance_components.csv"), index=False)

    print("CROSSED VARIANCE DECOMPOSITION + ICC (flaw-approval indicator)")
    print(vc_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    if logit.get("dev"):
        print(f"  admin/dev variance ratio: linear {comps['admin'] / comps['dev']:.1f}x, "
              f"logit {logit['admin'] / logit['dev']:.1f}x")

    print("\nMODEL-FREE CHECK (marginal cell means, persuasion condition)")
    marg = marginal_spread(df)
    for k, v in marg.items():
        print(f"  {k:10s} var {v['var']:.5f}  range {v['range']:.3f}")
    pd.DataFrame(marg).T.to_csv(
        os.path.join(args.output_dir, "marginal_spread.csv"))

    boot = clustered_bootstrap(df, n_boot=args.n_boot, seed=args.seed)
    pd.DataFrame([boot]).to_csv(
        os.path.join(args.output_dir, "clustered_inference.csv"), index=False)

    tw = boot["trace_ci"][1] - boot["trace_ci"][0]
    cw = boot["cluster_ci"][1] - boot["cluster_ci"][0]
    print("\nPERSUASION DELTA UNDER TWO RESAMPLING UNITS")
    print(f"  point estimate            {boot['point']:+.4f}")
    print(f"  trace-level 95% CI        [{boot['trace_ci'][0]:+.4f}, {boot['trace_ci'][1]:+.4f}]"
          f"  width {tw:.4f}")
    print(f"  scenario-clustered 95% CI [{boot['cluster_ci'][0]:+.4f}, {boot['cluster_ci'][1]:+.4f}]"
          f"  width {cw:.4f}")
    print(f"  inflation factor          {cw / tw:.2f}x")

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write("Experiment 1 crossed variance decomposition\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"traces: {len(df):,}   scenarios: {df.scenario_id.nunique()}\n\n")
        fh.write(vc_df.to_string(index=False) + "\n\n")
        fh.write(f"persuasion delta {boot['point']:+.4f}\n")
        fh.write(f"  trace-level 95% CI        [{boot['trace_ci'][0]:+.4f}, "
                 f"{boot['trace_ci'][1]:+.4f}] width {tw:.4f}\n")
        fh.write(f"  scenario-clustered 95% CI [{boot['cluster_ci'][0]:+.4f}, "
                 f"{boot['cluster_ci'][1]:+.4f}] width {cw:.4f}\n")
        fh.write(f"  inflation factor {cw / tw:.2f}x\n")

    write_latex(vc_df, boot, os.path.join(args.output_dir, "variance_table.tex"),
                len(df), df.scenario_id.nunique(), df.pair.nunique())
    print(f"\nwrote -> {args.output_dir}/")


if __name__ == "__main__":
    main()
