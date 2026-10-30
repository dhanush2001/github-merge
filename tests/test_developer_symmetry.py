"""A flawed PR and its clean twin must get identical developer treatment.

Experiments 1 and 2 gave the adversarial hints and the turn-by-turn escalation
script only to flawed PRs, so in the persuasion condition the developer's tone
alone could tell the admin which code was flawed. These tests build the real
developer prompts for every pair, every persuasion mode and every turn, and
fail if anything other than the commit itself differs.
"""
import glob
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.developer_agent import DEV_TRAP_HINTS, _build_messages  # noqa: E402
from config import CFG, PersuasionMode  # noqa: E402
from models import Scenario  # noqa: E402


def _pairs():
    by_id = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "dataset_b_*.json"))):
        for s in json.load(open(path)):
            by_id[s["scenario_id"]] = Scenario(**s)
    return [(t, by_id[sid.replace("_TRAP", "_CTRL")])
            for sid, t in by_id.items() if "_TRAP" in sid]


PAIRS = _pairs()


@pytest.fixture
def mode(request):
    saved = CFG.persuasion_mode
    CFG.persuasion_mode = request.param
    yield request.param
    CFG.persuasion_mode = saved


def _prompt(scenario, turn):
    feedback = "Please explain the change." if turn > 1 else ""
    messages, _ = _build_messages(scenario, feedback, [], turn)
    return "\n".join(m["content"] for m in messages).replace(scenario.developer_commit, "<COMMIT>")


def test_pairs_loaded():
    assert len(PAIRS) == 300


@pytest.mark.parametrize("mode", list(PersuasionMode), indirect=True, ids=lambda m: m.value)
@pytest.mark.parametrize("turn", [1, 2, 3, 4, 5, 6])
def test_flawed_and_clean_get_identical_developer_prompts(mode, turn):
    diffs = [t.scenario_id for t, c in PAIRS if _prompt(t, turn) != _prompt(c, turn)]
    assert not diffs, f"{len(diffs)} pairs differ in mode={mode.value} turn={turn}, e.g. {diffs[:3]}"


@pytest.mark.parametrize("mode", [PersuasionMode.FULL], indirect=True, ids=["full"])
def test_full_mode_escalates_and_hints_both_versions(mode):
    assert CFG.persuasion_mode is mode is PersuasionMode.FULL
    trap, ctrl = PAIRS[0]
    for s in (trap, ctrl):
        assert DEV_TRAP_HINTS in _prompt(s, 1)
        assert "Turn strategy:" in _prompt(s, 2)
