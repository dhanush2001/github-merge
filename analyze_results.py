import argparse
import glob
import json
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _latest_file(pattern: str) -> Optional[str]:
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    )


def _save_plot(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_decision_distribution(df: pd.DataFrame, out_dir: str) -> None:
    counts = df["final_decision"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Final Decision Distribution")
    ax.set_xlabel("Decision")
    ax.set_ylabel("Count")
    _save_plot(fig, os.path.join(out_dir, "decision_distribution.png"))


def _plot_approval_by_category(df: pd.DataFrame, out_dir: str) -> None:
    tmp = df.copy()
    tmp["approved"] = tmp["final_decision"].astype(str).eq("APPROVE")
    by_cat = tmp.groupby("category", as_index=False)["approved"].mean()
    by_cat = by_cat.sort_values("approved", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(by_cat["category"], by_cat["approved"])
    ax.set_ylim(0, 1)
    ax.set_title("Approval Rate by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Approval Rate")
    ax.tick_params(axis="x", rotation=30)
    _save_plot(fig, os.path.join(out_dir, "approval_rate_by_category.png"))


def _plot_avg_turns_by_category(df: pd.DataFrame, out_dir: str) -> None:
    by_cat = df.groupby("category", as_index=False)["total_turns"].mean()
    by_cat = by_cat.sort_values("total_turns", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(by_cat["category"], by_cat["total_turns"])
    ax.set_title("Average Turns by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Average Turns")
    ax.tick_params(axis="x", rotation=30)
    _save_plot(fig, os.path.join(out_dir, "avg_turns_by_category.png"))


def _plot_pairing_matrix(df: pd.DataFrame, out_dir: str) -> None:
    tmp = df.copy()
    tmp["approved"] = tmp["final_decision"].astype(str).eq("APPROVE")
    pivot = tmp.pivot_table(
        index="dev_model",
        columns="admin_model",
        values="approved",
        aggfunc="mean",
        fill_value=0.0,
    )

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot.index) * 0.8)))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Approval Rate by Dev/Admin Pairing")
    ax.set_xlabel("Admin Model")
    ax.set_ylabel("Dev Model")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Approval Rate")
    _save_plot(fig, os.path.join(out_dir, "pairing_approval_matrix.png"))


def _plot_survival_hist(df: pd.DataFrame, out_dir: str) -> None:
    if "dev_code_survival_rate" not in df.columns:
        return
    survival = pd.to_numeric(df["dev_code_survival_rate"], errors="coerce").dropna()
    if survival.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(survival, bins=10)
    ax.set_title("Developer Code Survival Rate Distribution")
    ax.set_xlabel("Survival Rate")
    ax.set_ylabel("Count")
    _save_plot(fig, os.path.join(out_dir, "survival_rate_hist.png"))


def _plot_confusion(df: pd.DataFrame, out_dir: str) -> None:
    if "expected_outcome" not in df.columns:
        return
    expected = df["expected_outcome"].astype(str)
    final = df["final_decision"].astype(str)

    valid = expected.isin(["APPROVE", "REJECT"])
    if valid.sum() == 0:
        return

    exp = expected[valid]
    pred = final[valid]

    matrix = pd.DataFrame(0, index=["APPROVE", "REJECT"], columns=["APPROVE", "REJECT"])
    for e, p in zip(exp, pred):
        if p in matrix.columns:
            matrix.loc[e, p] += 1

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix.values, vmin=0)
    ax.set_title("Expected vs Final Decision")
    ax.set_xlabel("Final")
    ax.set_ylabel("Expected")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix.values[i, j]), ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, label="Count")
    _save_plot(fig, os.path.join(out_dir, "decision_confusion_matrix.png"))


def _write_summary(df: pd.DataFrame, metrics_path: Optional[str], out_dir: str) -> None:
    approved = df["final_decision"].astype(str).eq("APPROVE").mean()
    timed_out = _coerce_bool(df.get("timed_out", pd.Series([False] * len(df)))).fillna(False).mean()
    unit_test_pass = _coerce_bool(df.get("unit_test_passed", pd.Series([False] * len(df)))).fillna(False).mean()

    lines = [
        "Evaluation Summary",
        "==================",
        f"Rows: {len(df)}",
        f"Approval rate: {approved:.3f}",
        f"Unit test pass rate: {unit_test_pass:.3f}",
        f"Timeout rate: {timed_out:.3f}",
        f"Average turns: {pd.to_numeric(df['total_turns'], errors='coerce').mean():.3f}",
        "",
        "Top categories by approval rate:",
    ]

    tmp = df.copy()
    tmp["approved"] = tmp["final_decision"].astype(str).eq("APPROVE")
    by_cat = tmp.groupby("category", as_index=False)["approved"].mean()
    by_cat = by_cat.sort_values("approved", ascending=False).head(5)
    for _, row in by_cat.iterrows():
        lines.append(f"- {row['category']}: {row['approved']:.3f}")

    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        screening = (
            metrics.get("dataset_b", {})
            .get("admin", {})
            .get("screening", {})
        )
        if screening:
            lines.extend(
                [
                    "",
                    "Dataset B screening:",
                    f"- Overall accuracy: {screening.get('overall_accuracy', 0.0):.3f}",
                    f"- True negative rate: {screening.get('true_negative_rate', 0.0):.3f}",
                    f"- False positive rate: {screening.get('false_positive_rate', 0.0):.3f}",
                ]
            )

    out_path = os.path.join(out_dir, "summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and visualize evaluation results.")
    parser.add_argument("--results-csv", default=None, help="Path to results_*.csv")
    parser.add_argument("--metrics-json", default=None, help="Path to metrics_*.json")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots and summary")
    args = parser.parse_args()

    results_csv = args.results_csv or _latest_file("results/results_*.csv")
    metrics_json = args.metrics_json or _latest_file("results/metrics_*.json")

    if not results_csv or not os.path.exists(results_csv):
        raise FileNotFoundError("Could not find results CSV. Run run_eval.py first or pass --results-csv.")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"results/analysis_{stamp}"

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(results_csv)

    _plot_decision_distribution(df, out_dir)
    _plot_approval_by_category(df, out_dir)
    _plot_avg_turns_by_category(df, out_dir)
    _plot_pairing_matrix(df, out_dir)
    _plot_survival_hist(df, out_dir)
    _plot_confusion(df, out_dir)
    _write_summary(df, metrics_json, out_dir)

    print(f"Results CSV: {results_csv}")
    if metrics_json:
        print(f"Metrics JSON: {metrics_json}")
    print(f"Analysis output: {out_dir}")


if __name__ == "__main__":
    main()
