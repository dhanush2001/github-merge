import json, os, argparse
from datetime import datetime
from itertools import product
from models import Scenario, ScenarioResult, AdminDecision, DatasetType
from pipeline.negotiation import run_negotiation
from pipeline.judge import judge_interaction
from pipeline.code_runner import detect_hallucinated_imports
from evaluation.metrics import compute_all_metrics, results_to_dataframe
from config import CFG, MODELS
import pandas as pd


def _extract_survival_rate(trace) -> float:
    value = getattr(trace, "_survival_rate", 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    nested = getattr(value, "survival_rate", 0.0)
    return float(nested) if isinstance(nested, (int, float)) else 0.0


def load_all_scenarios():
    all_scenarios = []
    enabled_entries = [entry for entry in CFG.datasets if entry.enabled]
    if len(enabled_entries) > 1:
        print(f"  [WARN] Multiple datasets enabled; using only: {enabled_entries[0].label}")
        enabled_entries = enabled_entries[:1]

    for entry in enabled_entries:
        if not entry.enabled:
            continue
        if not os.path.exists(entry.path):
            print(f"  [WARN] Skipping missing file: {entry.path}")
            continue
        with open(entry.path) as f:
            raw = json.load(f)
        scenarios = [Scenario(**s) for s in raw]
        for s in scenarios:
            s.__dict__["_source_label"] = entry.label
        all_scenarios.extend((s, entry.label) for s in scenarios)
        print(f"  Loaded {len(scenarios):>3} scenarios [{entry.label}]")
    return all_scenarios


def run_single(scenario, dev_model, admin_model, dataset_label) -> ScenarioResult:
    trace = run_negotiation(scenario, dev_model, admin_model)
    hallucinated, _ = detect_hallucinated_imports(scenario.developer_commit)

    judge_score = None
    if scenario.dataset_type == DatasetType.A:
        judge_score = judge_interaction(scenario, trace)

    is_correct = None
    if scenario.expected_outcome:
        is_correct = (trace.final_decision == scenario.expected_outcome)

    result = ScenarioResult(
        scenario_id=scenario.scenario_id,
        dataset_type=scenario.dataset_type,
        category=scenario.category,
        dev_model=dev_model,
        admin_model=admin_model,
        same_model_pairing=(dev_model == admin_model),
        final_decision=trace.final_decision,
        expected_outcome=scenario.expected_outcome,
        total_turns=trace.total_turns,
        total_dev_chars=trace.total_dev_chars,
        total_dev_tokens=getattr(trace, "total_dev_tokens", 0),
        total_admin_chars=getattr(trace, "total_admin_chars", 0),
        total_admin_tokens=getattr(trace, "total_admin_tokens", 0),
        total_tokens=getattr(trace, "total_tokens", 0),
        timed_out=trace.timed_out,
        unit_test_passed=getattr(trace, "_unit_test_passed", False),
        unit_test_output=getattr(trace, "_unit_test_output", ""),
        dev_code_survival_rate=_extract_survival_rate(trace),
        judge_score=judge_score,
        is_correct_decision=is_correct,
        dataset_label=dataset_label,
        turns=trace.turns,
    )
    result.__dict__["hallucinated_imports"] = hallucinated
    result.__dict__["assertions_passed"]    = getattr(trace, "_assertions_passed", 0)
    result.__dict__["assertions_total"]     = getattr(trace, "_assertions_total", 0)
    return result


def main(args):
    os.makedirs(CFG.results_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.datasets:
        if len(args.datasets) > 1:
            print(f"  [WARN] Multiple --datasets values provided; using only: {args.datasets[0]}")
            args.datasets = args.datasets[:1]
        for entry in CFG.datasets:
            entry.enabled = entry.label in args.datasets
    if args.dev_models:
        CFG.dev_models = args.dev_models
    if args.admin_models:
        CFG.admin_models = args.admin_models

    all_scenarios = load_all_scenarios()
    pairings = list(product(CFG.dev_models, CFG.admin_models))
    if args.cross_only:
        pairings = [(d, a) for d, a in pairings if d != a]

    all_results = []
    conversation_logs = []
    for dev_model, admin_model in pairings:
        print(f"\n  Dev: {dev_model}  |  Admin: {admin_model}")
        for scenario, label in all_scenarios:
            print(f"  [{scenario.scenario_id}]", end=" ")
            try:
                result = run_single(scenario, dev_model, admin_model, label)
                all_results.append(result)
                conversation_logs.append(
                    {
                        "scenario_id": result.scenario_id,
                        "dataset_label": result.dataset_label,
                        "dataset_type": str(result.dataset_type),
                        "category": result.category,
                        "dev_model": result.dev_model,
                        "admin_model": result.admin_model,
                        "final_decision": str(result.final_decision),
                        "total_turns": result.total_turns,
                        "turns": [t.model_dump() for t in result.turns],
                    }
                )
                status = "✓" if result.final_decision == AdminDecision.APPROVE else "✗"
                print(
                    f"{status} Turns:{result.total_turns} "
                    f"Tokens:{result.total_tokens} Tests:{result.unit_test_passed}"
                )
            except Exception as e:
                print(f"ERROR: {e}")

    out_json = f"{CFG.results_dir}/results_{run_id}.json"
    out_csv  = f"{CFG.results_dir}/results_{run_id}.csv"
    out_conversation_json = f"{CFG.results_dir}/conversation_logs_{run_id}.json"
    
    # Verify results directory exists before writing
    if not os.path.exists(CFG.results_dir):
        print(f"[ERROR] Results directory does not exist: {CFG.results_dir}")
        print(f"[ERROR] Current working directory: {os.getcwd()}")
        os.makedirs(CFG.results_dir, exist_ok=True)
        print(f"[INFO] Created directory: {CFG.results_dir}")
    
    with open(out_json, "w") as f:
        json.dump([r.model_dump(exclude={"turns"}) for r in all_results], f, indent=2, default=str)

    with open(out_conversation_json, "w") as f:
        json.dump(conversation_logs, f, indent=2, default=str)

    results_to_dataframe(all_results).to_csv(out_csv, index=False)

    metrics = compute_all_metrics(all_results)
    with open(f"{CFG.results_dir}/metrics_{run_id}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Results: {out_json}")
    print(f"Conversation logs: {out_conversation_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-only",    action="store_true")
    parser.add_argument("--datasets",      nargs="+", default=None)
    parser.add_argument("--dev-models",    nargs="+", default=None)
    parser.add_argument("--admin-models",  nargs="+", default=None)
    main(parser.parse_args())
