"""The admin (reviewer) must never see anything that reveals the ground truth.

Experiments 1 and 2 were run with ``### CATEGORY: {scenario.category}`` in the
admin prompt, which told reviewers "Security_Trap" vs "Control_Group". These
tests build the real prompts for every scenario in data/ and fail if a label
leaks back in.
"""
import glob
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.admin_agent import build_admin_messages  # noqa: E402
from models import Scenario  # noqa: E402

LABEL_PATTERNS = [
    r"_trap\b", r"\btrap\b", r"control[_ ]group", r"\bTRAP\b", r"\bCTRL\b",
    r"expected[_ ]outcome", r"PR_B_",
]


def _scenarios():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "dataset_b_*.json"))):
        out += [Scenario(**s) for s in json.load(open(path))]
    return out


SCENARIOS = _scenarios()


def _admin_text(scenario):
    messages, _ = build_admin_messages(scenario, dev_argument="Please merge.", turn=1)
    # The system prompt is identical for flawed and clean PRs, and legitimately
    # lists the flaw types as a review rubric; check only what varies per PR.
    return "\n".join(m["content"] for m in messages if m["role"] != "system")


def test_dataset_loaded():
    assert len(SCENARIOS) == 600


@pytest.mark.parametrize("pattern", LABEL_PATTERNS)
def test_admin_prompt_has_no_label(pattern):
    leaks = [s.scenario_id for s in SCENARIOS
             if re.search(pattern, _admin_text(s), re.IGNORECASE)]
    assert not leaks, f"{len(leaks)} admin prompts match {pattern!r}, e.g. {leaks[:3]}"


def test_admin_prompt_never_contains_category_or_id():
    for s in SCENARIOS:
        text = _admin_text(s)
        assert s.category not in text, s.scenario_id
        assert s.scenario_id not in text, s.scenario_id


def test_flawed_and_clean_twins_get_identical_framing():
    """A flawed PR and its clean twin may differ only in the commit itself."""
    by_id = {s.scenario_id: s for s in SCENARIOS}
    for sid, trap in by_id.items():
        if "_TRAP_" not in sid:
            continue
        ctrl = by_id[sid.replace("_TRAP_", "_CTRL_")]
        a = _admin_text(trap).replace(trap.developer_commit, "<COMMIT>")
        b = _admin_text(ctrl).replace(ctrl.developer_commit, "<COMMIT>")
        assert a == b, sid
