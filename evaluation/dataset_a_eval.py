import json, os
from datetime import datetime
from itertools import product
from typing import List
from models import Scenario, ScenarioResult, AdminDecision, DatasetType
from pipeline.negotiation import run_negotiation
from pipeline.judge import judge_interaction
from pipeline.code_runner import detect_hallucinated_imports
from evaluation.metrics import compute_developer_metrics, compute_admin_metrics, compute_judge_metrics, results_to_dataframe
from config import CFG


def _extract_survival_rate(trace) -> float:
    value = getattr(trace, "_survival_rate", 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    nested = getattr(value, "survival_rate", 0.0)
    return float(nested) if isinstance(nested, (int, float)) else 0.0


def load_dataset_a(paths: List[str]) -> List[Scenario]:
    scenarios = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [WARN] Not found: {path}")
            continue
        with open(path) as f:
            raw = json.load(f)
        loaded = [Scenario(**s) for s in raw if s.get("dataset_type") == DatasetType.A]
        print(f"  Loaded {len(loaded):>3} scenarios from {path}")
        scenarios.extend(loaded)
    return scenarios


def evaluate_scenario_a(scenario: Scenario, dev_model: str, admin_model: str) -> ScenarioResult:
    trace = run_negotiation(scenario, dev_model, admin_model)
    hallucinated, _ = detect_hallucinated_imports(scenario.developer_commit)
    judge_score = judge_interaction(scenario, trace)

    result = ScenarioResult(
        scenario_id=scenario.scenario_id,
        dataset_type=DatasetType.A,
        category=scenario.category,
        dev_model=dev_model,
        admin_model=admin_model,
        same_model_pairing=(dev_model == admin_model),
        final_decision=trace.final_decision,
        expected_outcome=None,
        total_turns=trace.total_turns,
        total_dev_chars=trace.total_dev_chars,
        timed_out=trace.timed_out,
        unit_test_passed=getattr(trace, "_unit_test_passed", False),
        unit_test_output=getattr(trace, "_unit_test_output", ""),
        dev_code_survival_rate=_extract_survival_rate(trace),
        judge_score=judge_score,
        is_correct_decision=None,
        dataset_label=getattr(scenario, "_source_label", "dataset_a"),
    )
    result.__dict__["hallucinated_imports"] = hallucinated
    result.__dict__["assertions_passed"]    = getattr(trace, "_assertions_passed", 0)
    result.__dict__["assertions_total"]     = getattr(trace, "_assertions_total", 0)
    result.__dict__["execution_time_ms"]    = getattr(trace, "_execution_time_ms", 0.0)
    return result


def run_dataset_a_evaluation(
    dataset_paths: List[str],
    run_id: str = None,
    cross_only: bool = False,
) -> List[ScenarioResult]:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(CFG.results_dir, exist_ok=True)

    scenarios = load_dataset_a(dataset_paths)
    if not scenarios:
        print("[ERROR] No Dataset A scenarios found.")
        return []

    pairings = list(product(CFG.dev_models, CFG.admin_models))
    if cross_only:
        pairings = [(d, a) for d, a in pairings if d != a]

    all_results = []
    print(f"\n  Dataset A: {len(scenarios)} scenarios × {len(pairings)} pairings\n")

    for dev_model, admin_model in pairings:
        print(f"\n  Dev: {dev_model}  |  Admin: {admin_model}")
        for scenario in scenarios:
            print(f"  [{scenario.scenario_id}]", end=" ")
            try:
                result = evaluate_scenario_a(scenario, dev_model, admin_model)
                all_results.append(result)
                status   = "✓" if result.final_decision == AdminDecision.APPROVE else "✗"
                survival = f"{result.dev_code_survival_rate:.0%}"
                print(f"{status} Survival:{survival} Turns:{result.total_turns}")
            except Exception as e:
                print(f"ERROR: {e}")

    out_json = f"{CFG.results_dir}/dataset_a_{run_id}.json"
    out_csv  = f"{CFG.results_dir}/dataset_a_{run_id}.csv"
    with open(out_json, "w") as f:
        json.dump([r.model_dump() for r in all_results], f, indent=2, default=str)
    results_to_dataframe(all_results).to_csv(out_csv, index=False)

    dev_m   = compute_developer_metrics(all_results)
    admin_m = compute_admin_metrics(all_results)
    judge_m = compute_judge_metrics(all_results)

    print(f"\n{'='*55}\n  DATASET A SUMMARY\n{'='*55}")
    print(f"  Persuasion Rate:     {dev_m.get('persuasion_rate', 0):.1%}")
    print(f"  Avg Turns (approve): {dev_m.get('avg_turns_on_approve', 0):.2f}")
    print(f"  Exec Pass Rate:      {admin_m.get('execution_pass_rate', 0):.1%}")
    if judge_m:
        print(f"  Manipulation Rate:   {judge_m.get('manipulation_rate', 0):.1%}")
        print(f"  Influence Delta:     {judge_m.get('avg_persuasion_influence_delta', 0):+.2f}")
    bias = admin_m.get("bias", {})
    if bias:
        print(f"  Bias Detected:       {bias.get('bias_detected')} (p={bias.get('chi2_p_value', 1.0):.4f})")
    print(f"\n  Results → {out_json}")
    return all_results
