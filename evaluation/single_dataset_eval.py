import json, os
from datetime import datetime
from typing import List
from models import Scenario, ScenarioResult, AdminDecision, DatasetType
from pipeline.negotiation import run_negotiation
from pipeline.code_runner import detect_hallucinated_imports
from pipeline.judge import judge_interaction
from evaluation.metrics import compute_all_metrics, results_to_dataframe
from config import CFG


def _extract_survival_rate(trace) -> float:
    value = getattr(trace, "_survival_rate", 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    nested = getattr(value, "survival_rate", 0.0)
    return float(nested) if isinstance(nested, (int, float)) else 0.0


def load_scenarios(paths: List[str]) -> List[Scenario]:
    scenarios = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [WARN] Not found: {path}")
            continue
        with open(path) as f:
            raw = json.load(f)
        loaded = [Scenario(**s) for s in raw]
        print(f"  Loaded {len(loaded):>3} scenarios from {path}")
        scenarios.extend(loaded)
    return scenarios


def evaluate_scenario(scenario: Scenario, dev_model: str, admin_model: str) -> ScenarioResult:
    trace = run_negotiation(scenario, dev_model, admin_model)
    hallucinated, _ = detect_hallucinated_imports(scenario.developer_commit)

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
        total_dev_input_tokens=trace.total_dev_input_tokens,
        total_dev_tokens=trace.total_dev_tokens,
        total_admin_chars=trace.total_admin_chars,
        total_admin_input_tokens=trace.total_admin_input_tokens,
        total_admin_tokens=trace.total_admin_tokens,
        total_input_tokens=trace.total_input_tokens,
        total_output_tokens=trace.total_output_tokens,
        total_tokens=trace.total_tokens,
        timed_out=trace.timed_out,
        unit_test_passed=getattr(trace, "_unit_test_passed", False),
        unit_test_output=getattr(trace, "_unit_test_output", ""),
        dev_code_survival_rate=_extract_survival_rate(trace),
        judge_score=judge_interaction(scenario, trace),
        is_correct_decision=is_correct,
        dataset_label=getattr(scenario, "_source_label", "dataset"),
        turns=trace.turns,
    )
    result.__dict__["hallucinated_imports"] = hallucinated
    result.__dict__["assertions_passed"] = getattr(trace, "_assertions_passed", 0)
    result.__dict__["assertions_total"]  = getattr(trace, "_assertions_total", 0)
    return result


def run_evaluation(
    dataset_paths: List[str],
    dev_models: List[str] = None,
    admin_models: List[str] = None,
    run_id: str = None,
    cross_only: bool = False,
) -> List[ScenarioResult]:
    run_id       = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    dev_models   = dev_models   or CFG.dev_models
    admin_models = admin_models or CFG.admin_models
    os.makedirs(CFG.results_dir, exist_ok=True)

    scenarios = load_scenarios(dataset_paths)
    if not scenarios:
        print("[ERROR] No scenarios found.")
        return []

    pairings = [(d, a) for d in dev_models for a in admin_models]
    if cross_only:
        pairings = [(d, a) for d, a in pairings if d != a]

    all_results = []
    total = len(scenarios) * len(pairings)
    print(f"\n  {len(scenarios)} scenarios × {len(pairings)} pairings = {total} tasks\n")

    for dev_model, admin_model in pairings:
        print(f"\n  Dev: {dev_model}  |  Admin: {admin_model}")
        for scenario in scenarios:
            try:
                result = evaluate_scenario(scenario, dev_model, admin_model)
                all_results.append(result)
                status = "✓" if result.final_decision == AdminDecision.APPROVE else "✗"
                print(f"  {status} {result.scenario_id} Turns:{result.total_turns} Tokens:{result.total_tokens}")
            except Exception as e:
                print(f"  ERROR {scenario.scenario_id}: {e}")

    out_json = f"{CFG.results_dir}/results_{run_id}.json"
    out_csv  = f"{CFG.results_dir}/results_{run_id}.csv"
    with open(out_json, "w") as f:
        json.dump([r.model_dump(exclude={"turns"}) for r in all_results], f, indent=2, default=str)
    results_to_dataframe(all_results).to_csv(out_csv, index=False)

    metrics = compute_all_metrics(all_results)
    with open(f"{CFG.results_dir}/metrics_{run_id}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n  Results → {out_json}")
    return all_results
