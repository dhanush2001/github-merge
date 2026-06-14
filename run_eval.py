import json, os, argparse, time
from datetime import datetime
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from models import Scenario, ScenarioResult, AdminDecision
from pipeline.negotiation import run_negotiation
from pipeline.judge import judge_interaction
from pipeline.code_runner import detect_hallucinated_imports
from evaluation.metrics import compute_all_metrics, results_to_dataframe
from config import CFG, MODELS, PersuasionMode
from checkpoint import CheckpointManager, TokenLimitReached, RateLimitReached, find_latest_checkpoint
import litellm
import pandas as pd

_print_lock = threading.Lock()

MAX_RETRIES = 3
_RETRY_BACKOFF = [60, 120, 180]

_RETRYABLE_ERRORS = (
    TokenLimitReached,
    RateLimitReached,
    litellm.exceptions.RateLimitError,
)

_MAJOR_ERRORS = (
    litellm.exceptions.AuthenticationError,
    litellm.exceptions.NotFoundError,
    litellm.exceptions.BadRequestError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.APIError,
)


def _to_json_native(value):
    """Recursively convert numpy/pandas scalars to JSON-native Python types."""
    if isinstance(value, dict):
        return {k: _to_json_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_native(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_native(v) for v in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value


def _extract_survival_rate(trace) -> float:
    value = getattr(trace, "_survival_rate", 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    nested = getattr(value, "survival_rate", 0.0)
    return float(nested) if isinstance(nested, (int, float)) else 0.0


def load_all_scenarios():
    all_scenarios = []
    enabled_entries = [entry for entry in CFG.datasets if entry.enabled]

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
        judge_score=judge_score,
        is_correct_decision=is_correct,
        dataset_label=dataset_label,
        persuasion_mode=CFG.persuasion_mode.value,
        turns=trace.turns,
    )
    result.__dict__["hallucinated_imports"] = hallucinated
    result.__dict__["assertions_passed"]    = getattr(trace, "_assertions_passed", 0)
    result.__dict__["assertions_total"]     = getattr(trace, "_assertions_total", 0)
    return result


def main(args):
    os.makedirs(CFG.results_dir, exist_ok=True)
    run_mode = CFG.persuasion_mode.value

    if getattr(args, "resume", None):
        resume_path = args.resume
        if resume_path == "auto":
            resume_path = find_latest_checkpoint(CFG.results_dir)
            if not resume_path:
                print("[ERROR] No checkpoint files found in results/. Starting fresh.")
                resume_path = None
        if resume_path:
            ckpt = CheckpointManager.from_file(resume_path)
            run_id = ckpt.run_id
            print(f"[INFO] Resuming checkpoint: {resume_path} (run_id={run_id})")
            print(f"[INFO] Checkpoint status: {ckpt.summary()}")
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt = CheckpointManager(CFG.results_dir, run_id)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt = CheckpointManager(CFG.results_dir, run_id)

    if args.datasets:
        requested_labels = set(args.datasets)
        known_labels = {entry.label for entry in CFG.datasets}
        unknown_labels = sorted(requested_labels - known_labels)
        if unknown_labels:
            print(f"  [WARN] Unknown dataset label(s): {', '.join(unknown_labels)}")
            print(f"  [INFO] Available labels: {', '.join(sorted(known_labels))}")
        requested_labels = set(args.datasets)
        known_labels = {entry.label for entry in CFG.datasets}
        unknown_labels = sorted(requested_labels - known_labels)
        if unknown_labels:
            print(f"  [WARN] Unknown dataset label(s): {', '.join(unknown_labels)}")
            print(f"  [INFO] Available labels: {', '.join(sorted(known_labels))}")
        for entry in CFG.datasets:
            entry.enabled = entry.label in requested_labels
            entry.enabled = entry.label in requested_labels
    if args.dev_models:
        CFG.dev_models = args.dev_models
    if args.admin_models:
        CFG.admin_models = args.admin_models

    all_scenarios = load_all_scenarios()
    if not all_scenarios:
        print("\n[ERROR] No scenarios loaded. Check --datasets labels or data/*.json files.")
        return
    if not all_scenarios:
        print("\n[ERROR] No scenarios loaded. Check --datasets labels or data/*.json files.")
        return
    pairings = list(product(CFG.dev_models, CFG.admin_models))
    if args.cross_only:
        pairings = [(d, a) for d, a in pairings if d != a]

    all_tasks = [
        (scenario, label, dev_model, admin_model)
        for dev_model, admin_model in pairings
        for scenario, label in all_scenarios
    ]
    total = len(all_tasks)
    completed = 0
    print(f"\n  Running {total} tasks across {len(pairings)} model pairing(s) "
          f"with {CFG.max_workers} workers...")

    # Seed results from checkpoint so metrics cover the full run, not just new tasks
    all_results = []
    for r in ckpt.all_completed_results():
        try:
            all_results.append(ScenarioResult(**r))
        except Exception:
            pass
    print(f"  Loaded {len(all_results)} completed results from checkpoint.")

    conversation_logs = [
        {
            "scenario_id": r.scenario_id,
            "run_mode": run_mode,
            "persuasion_mode": r.persuasion_mode,
            "dataset_label": r.dataset_label,
            "dataset_type": str(r.dataset_type),
            "category": r.category,
            "dev_model": r.dev_model,
            "admin_model": r.admin_model,
            "final_decision": str(r.final_decision),
            "total_turns": r.total_turns,
            "turns": [t.model_dump() for t in r.turns],
        }
        for r in all_results
    ]

    def _run_task(task):
        scenario, label, dev_model, admin_model = task
        if ckpt.is_completed(scenario.scenario_id, dev_model, admin_model):
            return None  # already done in a previous run

        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                return run_single(scenario, dev_model, admin_model, label)
            except _RETRYABLE_ERRORS as e:
                last_err = e
                wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                with _print_lock:
                    print(f"  [RETRY {attempt + 1}/{MAX_RETRIES}] {scenario.scenario_id} "
                          f"dev={dev_model} admin={admin_model} — {type(e).__name__}, "
                          f"waiting {wait}s...")
                time.sleep(wait)

        # All retries exhausted — raise so the caller skips without checkpointing
        raise last_err

    with ThreadPoolExecutor(max_workers=CFG.max_workers) as executor:
        future_to_task = {executor.submit(_run_task, task): task for task in all_tasks}
        for future in as_completed(future_to_task):
            scenario, label, dev_model, admin_model = future_to_task[future]
            try:
                result = future.result()
                if result is None:
                    # Skipped — already completed in prior run
                    with _print_lock:
                        completed += 1
                    continue
                status = "✓" if result.final_decision == AdminDecision.APPROVE else "✗"
                with _print_lock:
                    completed += 1
                    print(
                        f"  [{completed}/{total}] {status} {result.scenario_id} "
                        f"dev={dev_model} admin={admin_model} "
                        f"Turns:{result.total_turns} Tokens:{result.total_tokens} "
                        f"Tests:{result.unit_test_passed}"
                    )
                ckpt.mark_completed(
                    scenario.scenario_id, dev_model, admin_model,
                    result.model_dump(mode="json"),
                )
                all_results.append(result)
                conversation_logs.append(
                    {
                        "scenario_id": result.scenario_id,
                        "run_mode": run_mode,
                        "persuasion_mode": result.persuasion_mode,
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
            except (*_RETRYABLE_ERRORS, *_MAJOR_ERRORS) as e:
                # Rate/token limit (retries exhausted) or major API error — never checkpoint
                with _print_lock:
                    completed += 1
                    print(f"  [{completed}/{total}] SKIPPED (not checkpointed) "
                          f"{scenario.scenario_id} dev={dev_model} admin={admin_model}: "
                          f"{type(e).__name__}: {e}")
            except Exception as e:
                # Unexpected error — checkpoint so it's visible but doesn't block the run
                with _print_lock:
                    completed += 1
                    print(f"  [{completed}/{total}] ERROR {scenario.scenario_id} "
                          f"dev={dev_model} admin={admin_model}: {e}")
                ckpt.mark_error(scenario.scenario_id, dev_model, admin_model, str(e))

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
    metrics["run_meta"] = {
        "run_id": run_id,
        "run_mode": run_mode,
        "persuasion_enabled": CFG.persuasion_enabled,
        "temperature": CFG.persuasion_temperature if CFG.persuasion_enabled else CFG.control_temperature,
    }
    metrics = _to_json_native(metrics)
    with open(f"{CFG.results_dir}/metrics_{run_id}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Results: {out_json}")
    print(f"Conversation logs: {out_conversation_json}")


if __name__ == "__main__":
    valid_modes = [m.value for m in PersuasionMode]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-only",       action="store_true")
    parser.add_argument("--datasets",         nargs="+", default=None)
    parser.add_argument("--dev-models",       nargs="+", default=None)
    parser.add_argument("--admin-models",     nargs="+", default=None)
    parser.add_argument("--workers",          type=int,  default=None,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--resume",           nargs="?", const="auto", default=None,
                        metavar="CHECKPOINT_PATH",
                        help="Resume from a checkpoint. Omit path to auto-pick the latest.")
    parser.add_argument("--persuasion-mode",  default=None, choices=valid_modes,
                        help=(
                            "Ablation condition for developer agent persuasion. "
                            "control=neutral facts only | "
                            "rhetoric_only=persuasive framing | "
                            "hints_only=adversarial trap hints | "
                            "escalation_only=per-turn escalation templates | "
                            "full=all components (default)"
                        ))
    args = parser.parse_args()

    if args.persuasion_mode:
        CFG.persuasion_mode = PersuasionMode(args.persuasion_mode)

    if args.workers:
        CFG.max_workers = args.workers

    print(f"\n[INFO] Persuasion mode: {CFG.persuasion_mode.value}\n")
    main(args)
