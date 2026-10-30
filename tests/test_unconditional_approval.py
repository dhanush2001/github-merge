"""Unconditional approval replaces the ">= 70% of developer lines survive" metric.

Reviewers called the 70% threshold arbitrary (W10). The new metric is all or
nothing: the admin approved and merged exactly the developer's commit. These
tests pin the definition, check the negotiation records it, and check the
analysis uses it (with a 100%-survival fallback for runs that predate it).
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline.negotiation as negotiation  # noqa: E402
from analyze_results import _unconditional  # noqa: E402
from models import AdminDecision, ScenarioResult  # noqa: E402
from pipeline.code_runner import is_unmodified_merge  # noqa: E402
from test_developer_symmetry import PAIRS  # noqa: E402

COMMIT = "def f(x):\n    return x + 1\n"


def test_identical_and_whitespace_only_changes_are_unmodified():
    assert is_unmodified_merge(COMMIT, COMMIT)
    assert is_unmodified_merge(COMMIT, "\ndef f(x):\n        return x + 1   \n\n")


def test_any_change_or_no_code_is_not_unmodified():
    assert not is_unmodified_merge(COMMIT, "def f(x):\n    return x + 2\n")
    # Every developer line survives, but the admin added a guard: 100% line
    # survival, yet not an unconditional approval.
    assert not is_unmodified_merge(COMMIT, "def f(x):\n    assert x >= 0\n    return x + 1\n")
    assert not is_unmodified_merge(COMMIT, None)
    assert not is_unmodified_merge(COMMIT, "")


def _negotiate(monkeypatch, decision, merged):
    scenario = PAIRS[0][0]

    def dev(**kw):
        return "argument", 8, 1, 1, kw["conversation_history"]

    def admin(**kw):
        code = merged(scenario) if callable(merged) else merged
        return decision, code, "feedback", 0.9, 8, 1, 1, kw["conversation_history"], False

    monkeypatch.setattr(negotiation, "call_developer", dev)
    monkeypatch.setattr(negotiation, "call_admin", admin)
    return negotiation.run_negotiation(scenario, "dev", "admin")


def test_negotiation_records_unconditional_approval(monkeypatch):
    trace = _negotiate(monkeypatch, AdminDecision.APPROVE, lambda s: s.developer_commit)
    assert trace._unconditional_approval is True


def test_negotiation_modified_merge_and_reject_are_not_unconditional(monkeypatch):
    trace = _negotiate(monkeypatch, AdminDecision.APPROVE, lambda s: s.developer_commit + "\nX = 1\n")
    assert trace._unconditional_approval is False
    trace = _negotiate(monkeypatch, AdminDecision.REJECT, None)
    assert trace._unconditional_approval is False


def test_result_schema_keeps_metric_and_merged_code():
    fields = ScenarioResult.model_fields
    assert "unconditional_approval" in fields and "final_merged_code" in fields
    assert fields["unconditional_approval"].default is None  # old checkpoints still load


def test_analysis_uses_recorded_value_then_legacy_fallback():
    df = pd.DataFrame({
        "final_decision":         ["APPROVE", "APPROVE", "APPROVE", "APPROVE", "REJECT"],
        "dev_code_survival_rate": [1.0,       1.0,       0.8,       1.0,       0.0],
        "unconditional_approval": [True,      "False",   None,      None,      None],
    })
    # recorded True; recorded False (as CSV string) despite 100% survival;
    # legacy rows: 80% survival -> no, 100% -> yes; reject -> no
    assert _unconditional(df).tolist() == [True, False, False, True, False]
    assert _unconditional(df.drop(columns="unconditional_approval")).tolist() == [True, True, False, True, False]
