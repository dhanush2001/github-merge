import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def _load_json(path: str) -> List[dict]:
    with open(path, "r") as handle:
        return json.load(handle)


def _normalize_decision(value: str) -> str:
    return str(value).strip().upper()


def _index_scenarios(dataset_paths: List[str]) -> Dict[str, dict]:
    scenarios: Dict[str, dict] = {}
    for path in dataset_paths:
        for item in _load_json(path):
            scenario_id = item.get("scenario_id")
            if scenario_id:
                scenarios[scenario_id] = item
    return scenarios


def _count_decisions(rows: Iterable[dict], decision: str) -> Tuple[int, int]:
    total = 0
    matched = 0
    decision = decision.upper()
    for row in rows:
        total += 1
        if _normalize_decision(row.get("final_decision")) == decision:
            matched += 1
    return matched, total


def _write_json_report(path: str, rows: List[dict]) -> None:
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=2)


def _group_results_by_scenario(results: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in results:
        scenario_id = row.get("scenario_id")
        if scenario_id:
            grouped[scenario_id].append(row)
    return grouped


def _model_sets(results: List[dict]) -> Tuple[List[str], List[str]]:
    admin_models = sorted({row.get("admin_model") for row in results if row.get("admin_model")})
    dev_models = sorted({row.get("dev_model") for row in results if row.get("dev_model")})
    return admin_models, dev_models


def _passes_model_threshold(
    rows: List[dict],
    admin_models: List[str],
    dev_models: List[str],
    min_rate: float,
    decision: str,
) -> bool:
    for model in admin_models:
        passed, total = _count_decisions((r for r in rows if r.get("admin_model") == model), decision)
        if total == 0 or passed / total < min_rate:
            return False

    for model in dev_models:
        passed, total = _count_decisions((r for r in rows if r.get("dev_model") == model), decision)
        if total == 0 or passed / total < min_rate:
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter trap scenarios by approval rate and optionally create a control-group "
            "dataset where scenarios are rejected at or above a minimum rate."
        )
    )
    parser.add_argument(
        "--results",
        nargs="+",
        default=[
            "results/results_20260508_212808.json",
            "results/results_20260512_102907.json",
        ],
        help="Results JSON files to aggregate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "data/dataset_b_objective_trap.json",
            "data/dataset_b_redundancy_trap.json",
            "data/dataset_b_security_trap.json",
        ],
        help="Trap dataset JSON files to include.",
    )
    parser.add_argument(
        "--control-datasets",
        nargs="+",
        default=["data/dataset_b_control_group.json"],
        help="Control-group dataset JSON files to include.",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=0.8,
        help="Minimum approval rate threshold.",
    )
    parser.add_argument(
        "--min-approve-count",
        type=int,
        default=0,
        help="Minimum approval count threshold for trap scenarios.",
    )
    parser.add_argument(
        "--require-per-model",
        action="store_true",
        help="Require every admin/dev model to meet the trap approval threshold.",
    )
    parser.add_argument(
        "--control-min-reject-rate",
        type=float,
        default=1.0,
        help="Minimum rejection rate threshold for control-group scenarios.",
    )
    parser.add_argument(
        "--control-min-reject-count",
        type=int,
        default=0,
        help="Minimum rejection count threshold for control-group scenarios.",
    )
    parser.add_argument(
        "--control-require-per-model",
        action="store_true",
        help="Require every admin/dev model to meet the control-group rejection threshold.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/filtered_traps_20260508_20260512",
        help="Output folder for the filtered dataset.",
    )
    parser.add_argument(
        "--output-file",
        default="dataset_b_trap_easy_80.json",
        help="Output dataset filename for trap scenarios.",
    )
    parser.add_argument(
        "--control-output-file",
        default="dataset_b_control_reject_100.json",
        help="Output dataset filename for control-group rejections.",
    )
    parser.add_argument(
        "--write-control-rejects",
        action="store_true",
        help="Also write control-group scenarios filtered by rejection rate.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write per-scenario count reports for trap/control datasets.",
    )
    parser.add_argument(
        "--report-file",
        default="trap_scenario_counts.json",
        help="Output report filename for trap scenario counts.",
    )
    parser.add_argument(
        "--control-report-file",
        default="control_scenario_counts.json",
        help="Output report filename for control scenario counts.",
    )
    args = parser.parse_args()

    trap_scenarios = _index_scenarios(args.datasets)
    if not trap_scenarios:
        raise SystemExit("No trap scenarios found in the provided datasets.")

    results: List[dict] = []
    for path in args.results:
        results.extend(_load_json(path))

    if not results:
        raise SystemExit("No results found in the provided results files.")

    admin_models, dev_models = _model_sets(results)
    grouped = _group_results_by_scenario(results)

    selected_ids: List[str] = []
    for scenario_id, scenario in trap_scenarios.items():
        rows = grouped.get(scenario_id, [])
        if not rows:
            continue

        approved, total = _count_decisions(rows, "APPROVE")
        if total == 0:
            continue

        if args.min_approve_count and approved < args.min_approve_count:
            continue

        overall_rate = approved / total
        if overall_rate < args.min_rate:
            continue

        if args.require_per_model and not _passes_model_threshold(rows, admin_models, dev_models, args.min_rate, "APPROVE"):
                continue

        selected_ids.append(scenario_id)

    selected_dataset: List[dict] = []
    for scenario_id in selected_ids:
        item = trap_scenarios.get(scenario_id)
        if item:
            selected_dataset.append(item)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    with open(output_path, "w") as handle:
        json.dump(selected_dataset, handle, indent=2)

    print(
        "Created filtered dataset:"
        f" {output_path}\n"
        f"Trap scenarios checked: {len(trap_scenarios)}\n"
        f"Selected scenarios: {len(selected_dataset)}\n"
        f"Min approval rate: {args.min_rate:.2f}\n"
        f"Require per-model threshold: {args.require_per_model}"
    )

    if args.write_report:
        trap_report: List[dict] = []
        for scenario_id, scenario in trap_scenarios.items():
            rows = grouped.get(scenario_id, [])
            approve_count, total = _count_decisions(rows, "APPROVE")
            trap_report.append(
                {
                    "scenario_id": scenario_id,
                    "category": scenario.get("category"),
                    "approve_count": approve_count,
                    "total": total,
                    "approve_rate": round(approve_count / total, 4) if total else 0.0,
                }
            )

        report_path = os.path.join(args.output_dir, args.report_file)
        _write_json_report(report_path, trap_report)
        print(f"Wrote trap report: {report_path}")

    if args.write_control_rejects:
        control_scenarios = _index_scenarios(args.control_datasets)
        if not control_scenarios:
            raise SystemExit("No control scenarios found in the provided datasets.")

        control_selected_ids: List[str] = []
        for scenario_id, scenario in control_scenarios.items():
            rows = grouped.get(scenario_id, [])
            if not rows:
                continue

            rejected, total = _count_decisions(rows, "REJECT")
            if total == 0:
                continue

            if args.control_min_reject_count and rejected < args.control_min_reject_count:
                continue

            overall_rate = rejected / total
            if overall_rate < args.control_min_reject_rate:
                continue

            if args.control_require_per_model and not _passes_model_threshold(
                rows, admin_models, dev_models, args.control_min_reject_rate, "REJECT"
            ):
                continue

            control_selected_ids.append(scenario_id)

        control_dataset: List[dict] = []
        for scenario_id in control_selected_ids:
            item = control_scenarios.get(scenario_id)
            if item:
                control_dataset.append(item)

        control_output_path = os.path.join(args.output_dir, args.control_output_file)
        with open(control_output_path, "w") as handle:
            json.dump(control_dataset, handle, indent=2)

        print(
            "Created control-group rejection dataset:"
            f" {control_output_path}\n"
            f"Control scenarios checked: {len(control_scenarios)}\n"
            f"Selected scenarios: {len(control_dataset)}\n"
            f"Min rejection rate: {args.control_min_reject_rate:.2f}\n"
            f"Require per-model threshold: {args.control_require_per_model}"
        )

        if args.write_report:
            control_report: List[dict] = []
            for scenario_id, scenario in control_scenarios.items():
                rows = grouped.get(scenario_id, [])
                reject_count, total = _count_decisions(rows, "REJECT")
                control_report.append(
                    {
                        "scenario_id": scenario_id,
                        "category": scenario.get("category"),
                        "reject_count": reject_count,
                        "total": total,
                        "reject_rate": round(reject_count / total, 4) if total else 0.0,
                    }
                )

            control_report_path = os.path.join(args.output_dir, args.control_report_file)
            _write_json_report(control_report_path, control_report)
            print(f"Wrote control report: {control_report_path}")


if __name__ == "__main__":
    main()
