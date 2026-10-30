"""The admin cannot edit the commit: APPROVE merges exactly what was submitted.

Before Oct 2026 the admin returned its own merged_code and edited 23-36% of
approved flawed commits, so "approved" mixed "the flaw shipped" with
"approved but fixed while merging".
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.admin_agent import _sanitize_response, build_admin_messages  # noqa: E402
from models import AdminDecision  # noqa: E402
from pipeline.code_runner import is_unmodified_merge  # noqa: E402
from test_developer_symmetry import PAIRS  # noqa: E402

TRAP = PAIRS[0][0]


def test_prompt_no_longer_asks_for_merged_code_and_forbids_editing():
    system = build_admin_messages(TRAP, "argument", 1)[0][0]["content"]
    assert "merged_code" not in system
    assert "You cannot edit the code" in system


def test_returned_code_is_ignored_on_approve():
    payload = {"decision": "APPROVE", "merged_code": "def fixed(): pass", "feedback": "ok"}
    decision, merged, _, _ = _sanitize_response(payload, TRAP)
    assert decision == AdminDecision.APPROVE
    assert merged == TRAP.developer_commit
    assert is_unmodified_merge(TRAP.developer_commit, merged)


def test_no_code_unless_approved():
    for d in ("REJECT", "CLARIFY"):
        _, merged, _, _ = _sanitize_response({"decision": d, "merged_code": "x = 1"}, TRAP)
        assert merged is None
