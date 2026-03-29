# pipeline/negotiation.py
from models import Scenario, NegotiationTurn, NegotiationTrace, AdminDecision
from agents.developer_agent import call_developer
from agents.admin_agent import call_admin
from pipeline.code_runner import run_unit_tests, compute_code_survival_rate
from config import CFG
import time

def run_negotiation(scenario: Scenario, dev_model: str, admin_model: str) -> NegotiationTrace:
    turns = []
    # Maintain separate histories to prevent hint leakage
    dev_history = []
    admin_history = []
    admin_feedback = ""
    final_merged_code = None
    timed_out = False
    decision = None

    # Trackers for new token/char counts
    total_dev_chars = 0
    total_dev_tokens = 0
    total_admin_chars = 0
    total_admin_tokens = 0

    for turn_num in range(1, CFG.max_turns + 1):
        # Step 1: Developer argues
        dev_argument, dev_chars, dev_tokens, dev_history = call_developer(
            scenario=scenario, 
            model_key=dev_model, 
            conversation_history=dev_history, 
            admin_feedback=admin_feedback,
            turn=turn_num
        )
        
        total_dev_chars += dev_chars
        total_dev_tokens += dev_tokens

        # Step 2: Admin reviews
        decision, merged_code, admin_feedback, confidence, admin_chars, admin_tokens, admin_history = call_admin(
            scenario=scenario, 
            model_key=admin_model, 
            dev_argument=dev_argument, 
            turn=turn_num,
            conversation_history=admin_history
        )
        
        total_admin_chars += admin_chars
        total_admin_tokens += admin_tokens

        turn = NegotiationTurn(
            turn=turn_num,
            dev_argument=dev_argument,
            dev_char_count=dev_chars,
            dev_token_count=dev_tokens,
            admin_decision=decision,
            admin_feedback=admin_feedback,
            admin_char_count=admin_chars,
            admin_token_count=admin_tokens,
        )             
        turns.append(turn)

        # Routing logic:
        # - APPROVE ends negotiation immediately.
        # - CLARIFY trigger another persuasion turn until max_turns.
        if decision == AdminDecision.APPROVE:
            final_merged_code = merged_code
            break
        elif decision == AdminDecision.REJECT:
            final_merged_code = None
            break
            
    else:
        # If no approval within max_turns, finalize as Timeedout 
        timed_out = True
        decision = AdminDecision.TIMEOUT

    # Run unit tests on final merged code (or base+commit if no merge)
    test_code = final_merged_code or scenario.developer_commit
    test_result = run_unit_tests(test_code, scenario.unit_tests)
    unit_test_passed = test_result.passed
    unit_test_output = test_result.output if test_result.output else test_result.error
    
    survival_result = compute_code_survival_rate(scenario.developer_commit, final_merged_code or "")

    trace = NegotiationTrace(
        scenario_id=scenario.scenario_id,
        dev_model=dev_model,
        admin_model=admin_model,
        turns=turns,
        final_decision=decision,
        final_merged_code=final_merged_code,
        total_dev_chars=sum(t.dev_char_count for t in turns),
        total_dev_tokens=sum(t.dev_token_count for t in turns),
        total_admin_chars=sum(t.admin_char_count for t in turns),
        total_admin_tokens=sum(t.admin_token_count for t in turns),
        total_tokens=sum(t.dev_token_count + t.admin_token_count for t in turns),
        total_turns=len(turns),
        timed_out=timed_out,
    )
    # Attach test results to trace for downstream evaluation
    trace._unit_test_passed = unit_test_passed
    trace._unit_test_output = unit_test_output
    trace._survival_rate = survival_result.survival_rate
    trace._survival_result = survival_result
    trace._assertions_passed = test_result.assertions_passed
    trace._assertions_total = test_result.assertions_total

    return trace
