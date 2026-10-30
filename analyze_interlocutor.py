"""
Interlocutor-anchored full-vs-control analysis.

Motivation
----------
The generic ``analyze_results.py`` full-vs-control comparison silently
``fillna(0)``s any admin/dev cell that exists in one persuasion mode but not the
other.  When the control run only covers a single anchor model (e.g. gpt-5.1 as
admin) while the full run covers every model, that produces fabricated 0.0
control rates and meaningless deltas for every non-anchored model.

This script instead treats one model as a fixed *interlocutor anchor* and builds
two role slices:

  * anchor-as-DEV   : dev_model == anchor, admin_model varies
                      -> "how well does each admin resist the anchor's arguments?"
  * anchor-as-ADMIN : admin_model == anchor, dev_model varies
                      -> "how persuadable is the anchor screener against each dev?"

Within each slice it compares full vs control persuasion mode, but *only* on the
(model, scenario) cells that are present in BOTH modes.  Cells covered by one
mode only are reported as coverage gaps rather than folded into a delta.  Paired
McNemar exact tests run over matched (scenario_id, dev_model, admin_model) keys.

Usage
-----
    python analyze_interlocutor.py --anchor gpt-5.1 \
        --full results/full_20260711_232546 results/full_20260711_193227 \
        --control results/control_20260712_083023 results/control_20260712_075058 \
        --output-dir results/analysis_interlocutor_gpt51

Each --full / --control argument may be a results.csv file or a run directory
containing results.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

# Reuse the vetted stats/label helpers from the main analyzer so the two scripts
# stay consistent (same FPR/TPR/McNemar definitions).
from analyze_results import (
    _approved,
    _bootstrap_delta_ci,
    _bootstrap_mean_ci,
    _coerce_bool,
    _control_df,
    _mcnemar_exact,
    _model_color_map,
    _save_plot,
    _trap_df,
)

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

KEY_COLS = ["scenario_id", "dev_model", "admin_model"]


def _slice_label(slice_name: str) -> str:
    """Human-readable slice label for figure titles (no 'anchor'/'interlocutor')."""
    return {
        "anchor_as_dev": "dev-role (admins vary)",
        "anchor_as_admin": "admin-role (devs vary)",
    }.get(slice_name, slice_name.replace("_", " "))


# ── loading ─────────────────────────────────────────────────────────────────

def _resolve_csv(path: str) -> str:
    """Accept either a results.csv file or a run directory containing one."""
    if os.path.isdir(path):
        candidate = os.path.join(path, "results.csv")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"No results.csv in directory: {path}")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Path not found: {path}")


def _load_mode(paths: list[str], expected_mode: str) -> pd.DataFrame:
    """Load + concat several runs of one persuasion mode, de-duplicating cells.

    Repeated runs of the same (scenario, dev, admin) cell are collapsed to a
    single row so identical re-runs do not inflate n / tighten CIs artificially.
    """
    frames = []
    for p in paths:
        csv = _resolve_csv(p)
        frame = pd.read_csv(csv)
        frames.append(frame)
        modes = sorted(frame.get("persuasion_mode", pd.Series(dtype=str)).dropna().unique())
        print(f"  loaded {csv}  (n={len(frame)}, modes={modes})")
    df = pd.concat(frames, ignore_index=True)

    if "persuasion_mode" in df.columns:
        present = sorted(df["persuasion_mode"].dropna().unique())
        if present and present != [expected_mode]:
            print(f"  [WARN] expected only '{expected_mode}' rows but found modes {present}; "
                  f"keeping rows where persuasion_mode == '{expected_mode}'")
            df = df[df["persuasion_mode"] == expected_mode].copy()

    before = len(df)
    df = df.drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  de-duplicated {dropped} repeated cell(s); {len(df)} unique cells remain")
    df["mode"] = expected_mode
    return df


# ── per-cell metrics ─────────────────────────────────────────────────────────

def _timeout_rate(sub: pd.DataFrame) -> float:
    if "timed_out" not in sub.columns or sub.empty:
        return float("nan")
    return float(_coerce_bool(sub["timed_out"]).fillna(False).mean())


def _cell_metrics(sub: pd.DataFrame, varying_col: str, model: str) -> dict:
    """Full screening panel for one varying-model cell.

    Ground truth: traps must be REJECTed, control commits must be APPROVEd.
    The positive/target class for recall & precision is *trap detection*
    (correctly rejecting a bad commit), matching the adversarial framing.
    """
    traps = _trap_df(sub)
    ctrls = _control_df(sub)
    fp_vals = _approved(traps).to_numpy(dtype=float) if not traps.empty else np.array([])  # trap approved = FP
    tp_vals = _approved(ctrls).to_numpy(dtype=float) if not ctrls.empty else np.array([])  # control approved = TP

    n_trap = int(fp_vals.size)
    n_ctrl = int(tp_vals.size)
    fp = int(fp_vals.sum()); tn = n_trap - fp        # traps approved / rejected
    tp = int(tp_vals.sum()); fn = n_ctrl - tp        # controls approved / rejected

    fpr, fpr_lo, fpr_hi = _bootstrap_mean_ci(fp_vals)
    tpr, tpr_lo, tpr_hi = _bootstrap_mean_ci(tp_vals)
    tnr = tn / n_trap if n_trap else float("nan")    # trap-detection recall (= 1 - FPR)
    fnr = fn / n_ctrl if n_ctrl else float("nan")
    total = n_trap + n_ctrl
    accuracy = (tn + tp) / total if total else float("nan")
    balanced_accuracy = np.nanmean([tnr, tpr]) if total else float("nan")

    # Trap-detection (positive = REJECT a trap) precision / recall / F1.
    precision_trap = tn / (tn + fn) if (tn + fn) else float("nan")
    recall_trap = tnr
    if precision_trap and recall_trap and not (np.isnan(precision_trap) or np.isnan(recall_trap)) \
            and (precision_trap + recall_trap) > 0:
        f1_trap = 2 * precision_trap * recall_trap / (precision_trap + recall_trap)
    else:
        f1_trap = float("nan")

    return {
        varying_col: model,
        "n_trap": n_trap, "n_control": n_ctrl, "n_total": total,
        "traps_approved_fp": fp, "traps_rejected_tn": tn,
        "controls_approved_tp": tp, "controls_rejected_fn": fn,
        "trap_fpr": fpr, "trap_fpr_ci_low": fpr_lo, "trap_fpr_ci_high": fpr_hi,
        "trap_tnr_recall": tnr,
        "control_tpr": tpr, "control_tpr_ci_low": tpr_lo, "control_tpr_ci_high": tpr_hi,
        "control_fnr": fnr,
        "accuracy": accuracy, "balanced_accuracy": balanced_accuracy,
        "trap_precision": precision_trap, "trap_recall": recall_trap, "trap_f1": f1_trap,
        "timeout_rate": _timeout_rate(sub),
    }


def _mode_table(df_mode: pd.DataFrame, varying_col: str) -> pd.DataFrame:
    rows = [
        _cell_metrics(df_mode[df_mode[varying_col] == m], varying_col, m)
        for m in sorted(df_mode[varying_col].dropna().unique())
    ]
    return pd.DataFrame(rows)


def _screening_lines(df_mode: pd.DataFrame, tbl: pd.DataFrame,
                     varying_col: str, mode_name: str) -> list[str]:
    """Formatted per-model + pooled screening panel for one mode."""
    if tbl.empty:
        return []
    lines = [
        f"  -- {mode_name.upper()} screening panel "
        f"(recall = trap-detection = 1-FPR; timeout over all rows) --",
        f"  {'model':<18} {'FPR':>6} {'TNR/rec':>7} {'TPR':>6} {'FNR':>6} "
        f"{'Acc':>6} {'BalAcc':>6} {'F1':>6} {'Timeout':>7} {'nTrap':>6} {'nCtrl':>6}",
        "  " + "-" * 92,
    ]

    def _fmt(v):
        return f"{v:.3f}" if pd.notna(v) else "  n/a"

    for _, r in tbl.iterrows():
        lines.append(
            f"  {str(r[varying_col]):<18} "
            f"{_fmt(r['trap_fpr']):>6} {_fmt(r['trap_tnr_recall']):>7} {_fmt(r['control_tpr']):>6} "
            f"{_fmt(r['control_fnr']):>6} {_fmt(r['accuracy']):>6} {_fmt(r['balanced_accuracy']):>6} "
            f"{_fmt(r['trap_f1']):>6} {_fmt(r['timeout_rate']):>7} "
            f"{int(r['n_trap']):>6} {int(r['n_control']):>6}"
        )
    pooled = _cell_metrics(df_mode, varying_col, "POOLED")
    lines += [
        "  " + "-" * 92,
        f"  {'POOLED':<18} "
        f"{_fmt(pooled['trap_fpr']):>6} {_fmt(pooled['trap_tnr_recall']):>7} {_fmt(pooled['control_tpr']):>6} "
        f"{_fmt(pooled['control_fnr']):>6} {_fmt(pooled['accuracy']):>6} {_fmt(pooled['balanced_accuracy']):>6} "
        f"{_fmt(pooled['trap_f1']):>6} {_fmt(pooled['timeout_rate']):>7} "
        f"{int(pooled['n_trap']):>6} {int(pooled['n_control']):>6}",
    ]
    return lines


def _paired_delta(full_traps: pd.DataFrame, ctrl_traps: pd.DataFrame) -> dict:
    """Paired full-vs-control FPR on the matched trap cells (McNemar + bootstrap)."""
    if full_traps.empty or ctrl_traps.empty:
        return {}
    f = full_traps.copy();  f["ap_full"] = _approved(f)
    c = ctrl_traps.copy();  c["ap_ctrl"] = _approved(c)
    merged = f[KEY_COLS + ["ap_full"]].merge(c[KEY_COLS + ["ap_ctrl"]], on=KEY_COLS, how="inner")
    if merged.empty:
        return {}
    stats = _mcnemar_exact(merged["ap_full"], merged["ap_ctrl"])
    delta, lo, hi = _bootstrap_delta_ci(
        merged["ap_full"].to_numpy(dtype=float),
        merged["ap_ctrl"].to_numpy(dtype=float),
        paired=True,
    )
    return {
        "paired_delta_fpr": delta,
        "paired_ci_low": lo,
        "paired_ci_high": hi,
        "n_pairs": len(merged),
        "mcnemar_exact_p": stats["mcnemar_exact_p"],
    }


# ── one role slice (anchor fixed in dev or admin) ────────────────────────────

def analyze_slice(df_full: pd.DataFrame, df_ctrl: pd.DataFrame,
                  anchor: str, anchor_col: str, varying_col: str,
                  slice_name: str, out_dir: str) -> list[str]:
    """Compare full vs control for one role slice. Returns summary text lines."""
    full = df_full[df_full[anchor_col] == anchor].copy()
    ctrl = df_ctrl[df_ctrl[anchor_col] == anchor].copy()

    lines = [
        "", "=" * 72,
        f"SLICE: {slice_name}   (anchor '{anchor}' fixed as {anchor_col}, {varying_col} varies)",
        "=" * 72,
        f"  full rows   : {len(full)}   models: {sorted(full[varying_col].dropna().unique())}",
        f"  control rows: {len(ctrl)}   models: {sorted(ctrl[varying_col].dropna().unique())}",
    ]
    if full.empty and ctrl.empty:
        lines.append("  -> anchor never appears in this role; slice skipped.")
        return lines

    full_tbl = _mode_table(full, varying_col)
    ctrl_tbl = _mode_table(ctrl, varying_col)
    if not full_tbl.empty:
        full_tbl.to_csv(os.path.join(out_dir, f"{slice_name}_full_metrics.csv"), index=False)
    if not ctrl_tbl.empty:
        ctrl_tbl.to_csv(os.path.join(out_dir, f"{slice_name}_control_metrics.csv"), index=False)

    # Screening panels (FPR / TPR / recall / accuracy / timeout ...) per mode.
    if not full.empty:
        lines += [""] + _screening_lines(full, full_tbl, varying_col, "full")
    if not ctrl.empty:
        lines += [""] + _screening_lines(ctrl, ctrl_tbl, varying_col, "control")

    full_models = set(full[varying_col].dropna().unique())
    ctrl_models = set(ctrl[varying_col].dropna().unique())
    both = sorted(full_models & ctrl_models)
    full_only = sorted(full_models - ctrl_models)
    ctrl_only = sorted(ctrl_models - full_models)

    # Coverage reporting — the whole point of this script.
    if full_only:
        lines.append(f"  [coverage] full-only {varying_col}s (no control counterpart, no delta): {full_only}")
    if ctrl_only:
        lines.append(f"  [coverage] control-only {varying_col}s (no full counterpart, no delta): {ctrl_only}")
    if not both:
        lines.append(f"  [coverage] no {varying_col} is covered by BOTH modes -> no valid full-vs-control delta.")
        _plot_mode_fpr(full_tbl, ctrl_tbl, varying_col, slice_name, anchor, out_dir)
        return lines

    # Aligned full-vs-control per varying model.
    full_traps = _trap_df(full)
    ctrl_traps = _trap_df(ctrl)
    records = []
    for m in both:
        f_row = full_tbl[full_tbl[varying_col] == m].iloc[0]
        c_row = ctrl_tbl[ctrl_tbl[varying_col] == m].iloc[0]
        rec = {
            varying_col: m,
            "full_fpr": f_row["trap_fpr"], "full_n_trap": int(f_row["n_trap"]),
            "control_fpr": c_row["trap_fpr"], "control_n_trap": int(c_row["n_trap"]),
            "delta_fpr_unpaired": f_row["trap_fpr"] - c_row["trap_fpr"],
            "full_tpr": f_row["control_tpr"], "control_tpr": c_row["control_tpr"],
            "delta_tpr": f_row["control_tpr"] - c_row["control_tpr"],
            "full_trap_recall": f_row["trap_tnr_recall"], "control_trap_recall": c_row["trap_tnr_recall"],
            "full_timeout": f_row["timeout_rate"], "control_timeout": c_row["timeout_rate"],
            "delta_timeout": f_row["timeout_rate"] - c_row["timeout_rate"],
        }
        rec.update(_paired_delta(
            full_traps[full_traps[varying_col] == m],
            ctrl_traps[ctrl_traps[varying_col] == m],
        ))
        records.append(rec)

    cmp_df = pd.DataFrame(records).sort_values("delta_fpr_unpaired", ascending=False)
    cmp_path = os.path.join(out_dir, f"{slice_name}_full_vs_control.csv")
    cmp_df.to_csv(cmp_path, index=False)

    # Pooled paired delta across all shared cells.
    pooled = _paired_delta(
        full_traps[full_traps[varying_col].isin(both)],
        ctrl_traps[ctrl_traps[varying_col].isin(both)],
    )

    lines += [
        f"  shared {varying_col}s compared: {both}",
        f"  {'model':<20} {'full_fpr':>8} {'ctrl_fpr':>8} {'Δ_fpr':>7} {'pairedΔ':>8} {'mcnemar_p':>11} {'n_pairs':>7}",
        "  " + "-" * 74,
    ]
    for _, r in cmp_df.iterrows():
        pd_delta = r.get("paired_delta_fpr", float("nan"))
        pval = r.get("mcnemar_exact_p", float("nan"))
        npairs = r.get("n_pairs", 0)
        lines.append(
            f"  {str(r[varying_col]):<20} {r['full_fpr']:>8.3f} {r['control_fpr']:>8.3f} "
            f"{r['delta_fpr_unpaired']:>7.3f} "
            f"{(pd_delta if pd.notna(pd_delta) else float('nan')):>8.3f} "
            f"{(pval if pd.notna(pval) else float('nan')):>11.3g} "
            f"{int(npairs) if pd.notna(npairs) else 0:>7}"
        )
    if pooled:
        lines += [
            "  " + "-" * 74,
            f"  POOLED over shared cells: paired Δ_fpr = {pooled['paired_delta_fpr']:+.3f} "
            f"[{pooled['paired_ci_low']:+.3f}, {pooled['paired_ci_high']:+.3f}]  "
            f"McNemar p = {pooled['mcnemar_exact_p']:.3g}  n_pairs = {pooled['n_pairs']}",
        ]

    _plot_full_vs_control(cmp_df, varying_col, slice_name, anchor, out_dir)
    _plot_full_vs_control_recall(cmp_df, varying_col, slice_name, anchor, out_dir)
    _plot_roc(cmp_df, varying_col, slice_name, anchor, out_dir)

    # ── analyze_results-style extras, scoped to this slice ────────────────────
    lines += _category_full_vs_control(full, ctrl, both, varying_col, slice_name, anchor, out_dir)
    for mode_name, df_mode in (("full", full), ("control", ctrl)):
        if not df_mode.empty:
            _plot_model_x_category_fpr(df_mode, varying_col, mode_name, slice_name, out_dir)
            _plot_confusion(df_mode, slice_name, mode_name, out_dir)
    _judge_summary(full, ctrl, varying_col, slice_name, anchor, out_dir)
    _export_latex(cmp_df, pooled, slice_name, varying_col, anchor, out_dir)
    return lines


# ── plots ────────────────────────────────────────────────────────────────────

def _plot_roc(cmp_df: pd.DataFrame, varying_col: str, slice_name: str,
              anchor: str, out_dir: str) -> None:
    """
    ROC operating-point plot for a role slice. Each varying model is a point at
    (FPR = flaw acceptance, TPR = control acceptance), averaged over the
    scenarios it faces against the fixed anchor. The no-persuasion point is
    hollow, the full-persuasion point is filled, and an arrow marks the shift.
    Because these are binary classifiers (one operating point each, no threshold
    to sweep), we plot the points and their shift rather than a fitted curve;
    the persuasion effect appears as the rightward migration of susceptible
    models toward the chance diagonal.
    """
    if cmp_df.empty:
        return
    needed = {"full_fpr", "full_tpr", "control_fpr", "control_tpr"}
    if not needed.issubset(cmp_df.columns):
        return

    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # Data extent over both conditions. When one model is held fixed (e.g. the
    # anchor as reviewer) the points cluster tightly and a full 0-1 ROC is
    # unreadable, so we zoom to the data and identify models with a color legend
    # instead of crowded inline labels. Spread-out slices keep the full 0-1 view.
    xs_all = pd.concat([cmp_df["full_fpr"], cmp_df["control_fpr"]]).dropna()
    ys_all = pd.concat([cmp_df["full_tpr"], cmp_df["control_tpr"]]).dropna()
    xmin, xmax = float(xs_all.min()), float(xs_all.max())
    ymin, ymax = float(ys_all.min()), float(ys_all.max())
    zoomed = max(xmax - xmin, ymax - ymin) < 0.55
    x_mid = (xmin + xmax) / 2 if zoomed else 0.55

    ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1, label="chance")
    ax.scatter([], [], s=80, color="0.35", label="persuasion (filled)")
    ax.scatter([], [], s=70, facecolors="none", edgecolors="0.35", label="no persuasion (hollow)")

    color_map = _model_color_map([r[varying_col] for _, r in cmp_df.iterrows()])
    for i, (_, r) in enumerate(cmp_df.iterrows()):
        color = color_map[r[varying_col]]
        # Shift arrow no-persuasion -> persuasion (drawn first so points sit on top).
        if pd.notna(r["control_fpr"]) and pd.notna(r["control_tpr"]):
            ax.scatter(r["control_fpr"], r["control_tpr"], s=70, facecolors="none",
                       edgecolors=color, linewidths=1.4, zorder=3)
            ax.annotate("", xy=(r["full_fpr"], r["full_tpr"]),
                        xytext=(r["control_fpr"], r["control_tpr"]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.8),
                        zorder=2)
        # Identify every model via the legend box (no crowded inline labels).
        ax.scatter(r["full_fpr"], r["full_tpr"], s=90, color=color, zorder=4,
                   label=str(r[varying_col]))

    if zoomed:
        ax.set_xlim(max(0, xmin - 0.05), min(1, xmax + 0.08))
        ax.set_ylim(max(0, ymin - 0.05), min(1, ymax + 0.08))
    else:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("FPR  (flaw acceptance $\\rightarrow$ worse)")
    ax.set_ylabel("TPR  (clean acceptance $\\rightarrow$ better)")
    ax.set_title(f"ROC operating points — {_slice_label(slice_name)}\n"
                 "filled = persuasion, hollow = no persuasion (arrow = shift)",
                 fontsize=14)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9,
              ncol=2, columnspacing=1.0, handletextpad=0.4)
    _save_plot(fig, os.path.join(out_dir, f"{slice_name}_roc.png"))


def _plot_full_vs_control(cmp_df: pd.DataFrame, varying_col: str,
                          slice_name: str, anchor: str, out_dir: str) -> None:
    if cmp_df.empty:
        return
    models = cmp_df[varying_col].tolist()
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.3), 5.5))
    ax.bar(x - width / 2, cmp_df["full_fpr"], width, label="Full persuasion", color="tomato")
    ax.bar(x + width / 2, cmp_df["control_fpr"], width, label="Control (neutral)", color="steelblue")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{_slice_label(slice_name)}: trap FPR — full vs control\n(only cells covered by both modes)")
    ax.set_xlabel(varying_col)
    ax.set_ylabel("Trap False Positive Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    for i, row in enumerate(cmp_df.itertuples(index=False)):
        ax.text(x[i] - width / 2, row.full_fpr + 0.01, f"{row.full_fpr:.2f}", ha="center", va="bottom", fontsize=12)
        ax.text(x[i] + width / 2, row.control_fpr + 0.01, f"{row.control_fpr:.2f}", ha="center", va="bottom", fontsize=12)
        delta = row.full_fpr - row.control_fpr
        ax.text(x[i], -0.07, f"Δ{delta:+.2f}", ha="center", va="top", fontsize=12,
                color="red" if delta > 0 else "green", transform=ax.get_xaxis_transform())
    fig.subplots_adjust(bottom=0.2)
    _save_plot(fig, os.path.join(out_dir, f"{slice_name}_full_vs_control_fpr.png"))


def _plot_full_vs_control_recall(cmp_df: pd.DataFrame, varying_col: str,
                                 slice_name: str, anchor: str, out_dir: str) -> None:
    """Trap-detection recall (= 1 - FPR) per varying model, full vs control.
    Higher recall = more flawed commits correctly rejected."""
    if cmp_df.empty or "full_trap_recall" not in cmp_df.columns:
        return
    models = cmp_df[varying_col].tolist()
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.3), 5.5))
    ax.bar(x - width / 2, cmp_df["full_trap_recall"], width, label="Full persuasion", color="seagreen")
    ax.bar(x + width / 2, cmp_df["control_trap_recall"], width, label="Control (neutral)", color="steelblue")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{_slice_label(slice_name)}: trap-detection recall — full vs control\n"
                 f"(recall = 1 - FPR; only cells covered by both modes)")
    ax.set_xlabel(varying_col)
    ax.set_ylabel("Trap-Detection Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    for i, row in enumerate(cmp_df.itertuples(index=False)):
        ax.text(x[i] - width / 2, row.full_trap_recall + 0.01, f"{row.full_trap_recall:.2f}",
                ha="center", va="bottom", fontsize=12)
        ax.text(x[i] + width / 2, row.control_trap_recall + 0.01, f"{row.control_trap_recall:.2f}",
                ha="center", va="bottom", fontsize=12)
        delta = row.full_trap_recall - row.control_trap_recall
        ax.text(x[i], -0.07, f"Δ{delta:+.2f}", ha="center", va="top", fontsize=12,
                color="green" if delta > 0 else "red", transform=ax.get_xaxis_transform())
    fig.subplots_adjust(bottom=0.2)
    _save_plot(fig, os.path.join(out_dir, f"{slice_name}_full_vs_control_recall.png"))


def _plot_mode_fpr(full_tbl: pd.DataFrame, ctrl_tbl: pd.DataFrame,
                   varying_col: str, slice_name: str, anchor: str, out_dir: str) -> None:
    """Fallback single-mode FPR bar chart when the modes share no cells."""
    tbl = full_tbl if not full_tbl.empty else ctrl_tbl
    mode = "full" if not full_tbl.empty else "control"
    if tbl.empty:
        return
    tbl = tbl.sort_values("trap_fpr", ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(tbl) * 1.2), 5))
    ax.bar(tbl[varying_col].astype(str), tbl["trap_fpr"], color="tomato")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{_slice_label(slice_name)}: trap FPR ({mode} mode only — no shared cells for a delta)")
    ax.set_xlabel(varying_col)
    ax.set_ylabel("Trap False Positive Rate")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(tbl["trap_fpr"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    _save_plot(fig, os.path.join(out_dir, f"{slice_name}_{mode}_only_fpr.png"))


# ── analyze_results-style breakdowns (per slice) ─────────────────────────────

def _cat_fpr(df_mode: pd.DataFrame) -> pd.Series:
    """Trap FPR per category for a mode subset (trap categories only)."""
    traps = _trap_df(df_mode)
    if traps.empty:
        return pd.Series(dtype=float)
    t = traps.copy()
    t["fp"] = _approved(t)
    return t.groupby("category")["fp"].mean()


def _category_full_vs_control(full: pd.DataFrame, ctrl: pd.DataFrame, both: list[str],
                              varying_col: str, slice_name: str, anchor: str,
                              out_dir: str) -> list[str]:
    """Per trap-category FPR, full vs control + delta. CSV + grouped bar + summary lines."""
    # Restrict to models covered by both modes so the comparison is apples-to-apples.
    f = full[full[varying_col].isin(both)] if both else full
    c = ctrl[ctrl[varying_col].isin(both)] if both else ctrl
    full_fpr = _cat_fpr(f).rename("full_fpr")
    ctrl_fpr = _cat_fpr(c).rename("control_fpr")
    if full_fpr.empty and ctrl_fpr.empty:
        return []
    table = pd.concat([full_fpr, ctrl_fpr], axis=1)
    table["delta"] = table["full_fpr"] - table["control_fpr"]
    table = table.sort_values("delta", ascending=False)
    table.index.name = "category"
    table.to_csv(os.path.join(out_dir, f"{slice_name}_fpr_by_category.csv"))

    cats = table.index.tolist()
    x = np.arange(len(cats))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.6), 5))
    ax.bar(x - width / 2, table["full_fpr"], width, label="Full", color="tomato")
    ax.bar(x + width / 2, table["control_fpr"], width, label="Control", color="steelblue")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{_slice_label(slice_name)}: trap FPR by category — full vs control")
    ax.set_xlabel("Trap category")
    ax.set_ylabel("Trap False Positive Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.legend()
    for i, cat in enumerate(cats):
        d = table.loc[cat, "delta"]
        ax.text(x[i], -0.07, f"Δ{d:+.2f}", ha="center", va="top", fontsize=12,
                color="red" if d > 0 else "green", transform=ax.get_xaxis_transform())
    fig.subplots_adjust(bottom=0.2)
    _save_plot(fig, os.path.join(out_dir, f"{slice_name}_fpr_by_category.png"))

    lines = ["", "  -- trap FPR by category (full vs control, shared cells) --",
             f"  {'category':<22} {'full':>6} {'ctrl':>6} {'delta':>7}"]
    for cat in cats:
        r = table.loc[cat]
        lines.append(f"  {str(cat):<22} {r['full_fpr']:>6.3f} {r['control_fpr']:>6.3f} {r['delta']:>+7.3f}")
    return lines


def _plot_model_x_category_fpr(df_mode: pd.DataFrame, varying_col: str, mode: str,
                               slice_name: str, out_dir: str) -> None:
    """Heatmap: varying model × trap category FPR for one mode."""
    traps = _trap_df(df_mode)
    if traps.empty:
        return
    t = traps.copy()
    t["fp"] = _approved(t)
    pivot = t.pivot_table(index=varying_col, columns="category", values="fp",
                          aggfunc="mean", fill_value=float("nan"))
    counts = t.pivot_table(index=varying_col, columns="category", values="fp",
                           aggfunc="count", fill_value=0)
    if pivot.empty:
        return
    nrows, ncols = pivot.shape
    fig, ax = plt.subplots(figsize=(max(8, ncols * 2.2), max(4, nrows * 0.7 + 2)))
    im = ax.imshow(pivot.values.astype(float), aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn_r")
    ax.set_title(f"{_slice_label(slice_name)} [{mode}]: trap FPR by {varying_col} × category")
    ax.set_xlabel("Trap category")
    ax.set_ylabel(varying_col)
    ax.set_xticks(range(ncols)); ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=12)
    ax.set_yticks(range(nrows)); ax.set_yticklabels(pivot.index, fontsize=12)
    for i in range(nrows):
        for j in range(ncols):
            v = pivot.values[i, j]
            n = int(counts.values[i, j]) if i < counts.shape[0] and j < counts.shape[1] else 0
            txt = f"{v:.2f}\n(n={n})" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                    color="white" if (not np.isnan(v) and v > 0.7) else "black")
    fig.colorbar(im, ax=ax, label="FPR", shrink=0.8)
    _save_plot(fig, os.path.join(out_dir, f"{slice_name}_{mode}_fpr_{varying_col}_x_category.png"))
    pivot.to_csv(os.path.join(out_dir, f"{slice_name}_{mode}_fpr_{varying_col}_x_category.csv"))


def _plot_confusion(df_mode: pd.DataFrame, slice_name: str, mode: str, out_dir: str) -> None:
    """Expected vs final decision confusion matrix for one mode."""
    from analyze_results import _norm_label
    exp = _norm_label(df_mode["expected_outcome"])
    pred = _norm_label(df_mode["final_decision"])
    valid = exp.isin(["APPROVE", "REJECT"])
    if valid.sum() == 0:
        return
    exp, pred = exp[valid], pred[valid]
    matrix = pd.DataFrame(0, index=["APPROVE", "REJECT"], columns=["APPROVE", "REJECT"])
    for e, p in zip(exp, pred):
        if p in matrix.columns:
            matrix.loc[e, p] += 1
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(matrix.values, vmin=0, cmap="Blues")
    ax.set_title(f"{_slice_label(slice_name)} [{mode}]\nExpected vs Final decision",
                 fontsize=14, pad=12)
    ax.set_xlabel("Final"); ax.set_ylabel("Expected")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(matrix.columns); ax.set_yticklabels(matrix.index)
    for i in range(2):
        for j in range(2):
            val = matrix.values[i, j]
            ax.text(j, i, str(val), ha="center", va="center", fontsize=12,
                    color="white" if val > matrix.values.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="Count", shrink=0.85, pad=0.04)
    _save_plot(fig, os.path.join(out_dir, f"{slice_name}_{mode}_confusion.png"))


JUDGE_COLS = [
    "judge_persuasion_score", "judge_code_quality_score",
    "judge_admin_judgement_accuracy", "judge_persuasion_influence_delta",
    "judge_admin_rigor_score",
]


def _judge_summary(full: pd.DataFrame, ctrl: pd.DataFrame, varying_col: str,
                   slice_name: str, anchor: str, out_dir: str) -> None:
    """Mean judge scores per varying model per mode. CSV + pooled full-vs-control bar."""
    available = [c for c in JUDGE_COLS if c in full.columns or c in ctrl.columns]
    if not available:
        return
    rows = []
    for mode_name, df_mode in (("full", full), ("control", ctrl)):
        if df_mode.empty:
            continue
        tmp = df_mode.copy()
        for c in available:
            tmp[c] = pd.to_numeric(tmp.get(c), errors="coerce")
        for m, g in tmp.groupby(varying_col):
            rec = {"mode": mode_name, varying_col: m}
            rec.update({c: g[c].mean() for c in available})
            rows.append(rec)
    if not rows:
        return
    judge_df = pd.DataFrame(rows)
    judge_df.to_csv(os.path.join(out_dir, f"{slice_name}_judge_scores.csv"), index=False)

    # Pooled full-vs-control means.
    pooled = judge_df.groupby("mode")[available].mean()
    if pooled.empty:
        return
    labels = [c.replace("judge_", "") for c in available]
    x = np.arange(len(available))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(available) * 1.6), 5))
    if "full" in pooled.index:
        ax.bar(x - width / 2, pooled.loc["full", available].values, width, label="Full", color="tomato")
    if "control" in pooled.index:
        ax.bar(x + width / 2, pooled.loc["control", available].values, width, label="Control", color="steelblue")
    ax.set_title(f"{_slice_label(slice_name)}: mean judge scores — full vs control")
    ax.set_ylabel("Mean score")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=12)
    ax.legend()
    _save_plot(fig, os.path.join(out_dir, f"{slice_name}_judge_scores.png"))


def _export_latex(cmp_df: pd.DataFrame, pooled: dict, slice_name: str,
                  varying_col: str, anchor: str, out_dir: str) -> None:
    """Booktabs LaTeX table of the full-vs-control FPR comparison for this slice."""
    if cmp_df.empty:
        return

    def _f(v, d=3):
        return r"\texttt{n/a}" if pd.isna(v) else f"{v:.{d}f}"

    lines = [
        r"% Auto-generated by analyze_interlocutor.py — requires \usepackage{booktabs}",
        r"\begin{table}[ht]", r"\centering",
        rf"\caption{{Full vs.\ control trap FPR with anchor {anchor} fixed "
        rf"({slice_name.replace('_', ' ')}). Paired $\Delta$ and McNemar exact $p$ over matched cells.}}",
        rf"\label{{tab:{slice_name}}}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        rf"{varying_col.replace('_', ' ')} & Full FPR & Control FPR & $\Delta$ FPR & Paired $\Delta$ & McNemar $p$ \\",
        r"\midrule",
    ]
    for _, r in cmp_df.iterrows():
        lines.append(
            f"{str(r[varying_col]).replace('_', chr(92)+'_')} & {_f(r['full_fpr'])} & {_f(r['control_fpr'])} & "
            f"{_f(r['delta_fpr_unpaired'])} & {_f(r.get('paired_delta_fpr'))} & "
            f"{_f(r.get('mcnemar_exact_p'), 2) if pd.isna(r.get('mcnemar_exact_p')) else format(r.get('mcnemar_exact_p'), '.2g')} \\\\"
        )
    if pooled:
        lines += [
            r"\midrule",
            f"\\textbf{{Pooled}} & & & & {_f(pooled['paired_delta_fpr'])} & "
            f"{format(pooled['mcnemar_exact_p'], '.2g')} \\\\",
        ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(out_dir, f"{slice_name}_table.tex"), "w") as fh:
        fh.write("\n".join(lines))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", required=True, help="Interlocutor anchor model held fixed in each role.")
    ap.add_argument("--full", nargs="+", required=True, help="Full-mode results.csv files or run dirs.")
    ap.add_argument("--control", nargs="+", required=True, help="Control-mode results.csv files or run dirs.")
    ap.add_argument("--output-dir", default=None, help="Output directory (default: results/analysis_interlocutor_<anchor>).")
    args = ap.parse_args()

    print("Loading full runs:")
    df_full = _load_mode(args.full, "full")
    print("Loading control runs:")
    df_ctrl = _load_mode(args.control, "control")

    anchor = args.anchor
    seen_dev = set(df_full["dev_model"].unique()) | set(df_ctrl["dev_model"].unique())
    seen_admin = set(df_full["admin_model"].unique()) | set(df_ctrl["admin_model"].unique())
    if anchor not in seen_dev and anchor not in seen_admin:
        raise SystemExit(f"Anchor '{anchor}' not found as any dev_model or admin_model. "
                         f"Devs seen: {sorted(seen_dev)}  Admins seen: {sorted(seen_admin)}")

    out_dir = args.output_dir or f"results/analysis_interlocutor_{anchor.replace('.', '_').replace('/', '_')}"
    os.makedirs(out_dir, exist_ok=True)

    header = [
        "#" * 72,
        f"# INTERLOCUTOR-ANCHORED FULL vs CONTROL   —   anchor = {anchor}",
        f"# full cells: {len(df_full)}   control cells: {len(df_ctrl)}",
        "#" * 72,
    ]
    for h in header:
        print(h)

    lines = list(header)
    # anchor as DEV (interlocutor persuader) -> vary admin
    lines += analyze_slice(df_full, df_ctrl, anchor, "dev_model", "admin_model",
                           "anchor_as_dev", out_dir)
    # anchor as ADMIN (screener) -> vary dev
    lines += analyze_slice(df_full, df_ctrl, anchor, "admin_model", "dev_model",
                           "anchor_as_admin", out_dir)

    print("\n".join(lines[len(header):]))
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nOutput: {out_dir}/")
    for fname in sorted(os.listdir(out_dir)):
        print(f"  {fname}")


if __name__ == "__main__":
    main()
