#!/usr/bin/env python3
"""Scenario-clustered inference for the persuasion effect (Experiments 1 and 2).

Every admin x developer pairing reviews the SAME 300 flawed scenarios, so the
paired (control, full) outcomes are clustered by scenario: 25 pairings in
Experiment 1 and 9 candidates per slice in Experiment 2 share each scenario.
The plain McNemar test treats all of those pairs as independent and so
overstates significance. This script reports, per comparison:

  * plain McNemar (exact)                       -- what the paper used
  * clustered McNemar (Durkalski et al., 2003)  -- discordant pairs summed per
    scenario; variance from the between-scenario spread of (b_k - c_k)
  * design effect = plain chi2 / clustered chi2 (how much clustering inflated it)
  * GEE logistic regression, cluster = scenario (robust sandwich SE): odds ratio
    of approval under persuasion, 95% CI, p
  * crossed binomial GLMM (variational Bayes): random intercepts for scenario and
    pairing, fixed effects for persuasion, admin, developer and category; the
    persuasion odds ratio with a 95% credible interval

Only flawed commits (dataset_type B) enter: approval of a flawed commit is the
false-positive outcome the persuasion delta is defined on.

  python scripts/clustered_inference.py                 # both experiments
  python scripts/clustered_inference.py --no-glmm       # skip the slow GLMM
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from variance_decomposition import CELL_KEY, EXP1_MODELS, _timestamp, load_experiment1  # noqa: E402
from config import EXP2_ANCHORS, EXP2_CANDIDATES  # noqa: E402


def exp2_role(dev, admin, anchors=EXP2_ANCHORS, candidates=EXP2_CANDIDATES):
    """(anchor, slice, candidate) for an anchor x candidate pairing, else None."""
    if dev in anchors and admin in candidates:
        return dev, "candidate_screens", admin
    if admin in anchors and dev in candidates:
        return admin, "candidate_argues", dev
    return None


def load_experiment2(results_dir):
    """Deduplicated Experiment 2 flawed-commit traces with `anchor` and `slice` columns."""
    latest = {}
    paths = glob.glob(os.path.join(results_dir, "*_2026*", "conversation_logs.json"))
    for path in sorted(paths, key=lambda p: _timestamp(os.path.dirname(p))):
        with open(path) as fh:
            for rec in json.load(fh):
                latest[tuple(rec.get(k) for k in CELL_KEY)] = rec
    rows = []
    for rec in latest.values():
        if rec.get("dataset_type") != "DatasetType.B":
            continue
        dev, admin = rec["dev_model"], rec["admin_model"]
        role = exp2_role(dev, admin)
        if role is None:
            continue
        anchor, sl, candidate = role
        rows.append({
            "scenario_id": rec["scenario_id"], "admin": admin, "dev": dev,
            "persuasion": rec["persuasion_mode"], "category": rec.get("category"),
            "approved": int(rec.get("final_decision") == "AdminDecision.APPROVE"),
            "anchor": anchor, "slice": sl, "candidate": candidate,
        })
    df = pd.DataFrame(rows)
    df["pair"] = df["admin"] + "|" + df["dev"]
    return df


def paired(df):
    """One row per (scenario, admin, dev) cell observed in BOTH conditions."""
    wide = df.pivot_table(index=["scenario_id", "admin", "dev"], columns="persuasion",
                          values="approved", aggfunc="first").dropna(subset=["control", "full"])
    return wide.reset_index()


def mcnemar(wide):
    """Plain exact McNemar and the Durkalski clustered McNemar, cluster = scenario."""
    b = ((wide.full == 1) & (wide.control == 0)).astype(int)
    c = ((wide.full == 0) & (wide.control == 1)).astype(int)
    B, C = int(b.sum()), int(c.sum())
    p_plain = stats.binomtest(B, B + C, 0.5).pvalue if B + C else 1.0
    chi_plain = (B - C) ** 2 / (B + C) if B + C else 0.0

    per = pd.DataFrame({"s": wide.scenario_id, "d": b - c}).groupby("s").d.sum()
    denom = float((per ** 2).sum())
    chi_clust = (B - C) ** 2 / denom if denom else 0.0
    p_clust = float(stats.chi2.sf(chi_clust, 1))
    return {
        "n_pairs": len(wide), "n_scenarios": int(wide.scenario_id.nunique()),
        "fpr_control": float(wide.control.mean()), "fpr_full": float(wide.full.mean()),
        "delta": float(wide.full.mean() - wide.control.mean()),
        "b_full_only": B, "c_control_only": C,
        "chi2_plain": chi_plain, "p_plain_exact": float(p_plain),
        "chi2_clustered": chi_clust, "p_clustered": p_clust,
        "design_effect": chi_plain / chi_clust if chi_clust else float("nan"),
    }


def gee_or(df, fixed):
    """Population-averaged persuasion OR, scenario-clustered robust SE."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = df.sort_values("scenario_id").copy()
    d["pers"] = (d.persuasion == "full").astype(int)
    model = smf.gee(f"approved ~ pers + {fixed}", groups="scenario_id", data=d,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit()
    beta, se = res.params["pers"], res.bse["pers"]
    return {"gee_or": float(np.exp(beta)),
            "gee_ci_lo": float(np.exp(beta - 1.96 * se)), "gee_ci_hi": float(np.exp(beta + 1.96 * se)),
            "gee_p": float(res.pvalues["pers"])}


def glmm_or(df, fixed):
    """Crossed binomial GLMM (VB): random intercepts for scenario and pairing."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = df.copy()
    d["pers"] = (d.persuasion == "full").astype(int)
    vc = {"scenario": "0 + C(scenario_id)", "pairing": "0 + C(pair)"}
    model = BinomialBayesMixedGLM.from_formula(f"approved ~ pers + {fixed}", vc, d)
    res = model.fit_vb(verbose=False)
    i = model.exog_names.index("pers")
    m, s = float(res.fe_mean[i]), float(res.fe_sd[i])
    sds = dict(zip(model.vcp_names, np.exp(np.asarray(res.vcp_mean))))
    return {"glmm_or": float(np.exp(m)),
            "glmm_cri_lo": float(np.exp(m - 1.96 * s)), "glmm_cri_hi": float(np.exp(m + 1.96 * s)),
            "glmm_sd_scenario": float(sds.get("scenario", np.nan)),
            "glmm_sd_pairing": float(sds.get("pairing", np.nan))}


def analyse(label, df, fixed, run_glmm):
    out = {"comparison": label, **mcnemar(paired(df)), **gee_or(df, fixed)}
    if run_glmm:
        out.update(glmm_or(df, fixed))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output-dir", default="results/analysis_clustered")
    ap.add_argument("--no-glmm", action="store_true")
    args = ap.parse_args()
    glmm = not args.no_glmm
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    e1 = load_experiment1(args.results_dir)
    print(f"Exp1 flawed traces {len(e1):,}  scenarios {e1.scenario_id.nunique()}  pairings {e1.pair.nunique()}")
    rows.append(analyse("exp1_pooled", e1, "C(admin) + C(dev) + C(category)", glmm))
    for admin, g in e1.groupby("admin"):
        rows.append(analyse(f"exp1_admin={admin}", g, "C(dev) + C(category)", False))

    e2 = load_experiment2(args.results_dir)
    print(f"Exp2 flawed traces {len(e2):,}  scenarios {e2.scenario_id.nunique()}")
    for sl, g in e2.groupby("slice"):
        # Pooled over anchors (anchor as a fixed effect), then each anchor alone.
        if g.anchor.nunique() > 1:
            rows.append(analyse(f"exp2_{sl}", g, "C(anchor) + C(candidate) + C(category)", glmm))
        for anchor, ga in g.groupby("anchor"):
            rows.append(analyse(f"exp2_{sl}_anchor={anchor}", ga, "C(candidate) + C(category)", glmm))
            for cand, h in ga.groupby("candidate"):
                rows.append(analyse(f"exp2_{sl}_anchor={anchor}_{cand}", h, "C(category)", False))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.output_dir, "clustered_inference.csv"), index=False)

    cols = ["comparison", "n_pairs", "fpr_control", "fpr_full", "delta", "p_plain_exact",
            "p_clustered", "design_effect", "gee_or", "gee_ci_lo", "gee_ci_hi", "gee_p"]
    if glmm:
        cols += ["glmm_or", "glmm_cri_lo", "glmm_cri_hi", "glmm_sd_scenario", "glmm_sd_pairing"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        text = res[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}")
    print(text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(f"\nwrote -> {args.output_dir}/")


if __name__ == "__main__":
    main()
