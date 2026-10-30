"""SHORT_PERSUASIVE: FULL's persuasion at CONTROL's length (optional 2x2 cell).

With CONTROL (neutral, short), LENGTH_MATCHED (neutral, long) and FULL
(persuasive, long), this fills the persuasive-short cell so a tone x length
interaction can be tested. It must be FULL plus a word cap and nothing else.
Flawed/clean symmetry and sampling settings are covered for every mode by
tests/test_developer_symmetry.py and tests/test_sampling_settings.py.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.developer_agent import _build_messages  # noqa: E402
from config import CFG, PersuasionMode  # noqa: E402
from test_developer_symmetry import PAIRS  # noqa: E402

CAP = f" Use at most {CFG.short_persuasive_max_words} words."


def _messages(mode, scenario, turn, feedback=""):
    saved = CFG.persuasion_mode
    try:
        CFG.persuasion_mode = mode
        return _build_messages(scenario, feedback, [], turn)[0]
    finally:
        CFG.persuasion_mode = saved


@pytest.mark.parametrize("which", [0, 1])  # flawed and clean commit of a pair
@pytest.mark.parametrize("turn", [1, 2, 3, 6])
def test_is_full_plus_word_cap(which, turn):
    scenario = PAIRS[0][which]
    fb = "Why this change?" if turn > 1 else ""
    full = _messages(PersuasionMode.FULL, scenario, turn, fb)
    short = _messages(PersuasionMode.SHORT_PERSUASIVE, scenario, turn, fb)
    assert short[:-1] == full[:-1]  # same system prompt (rhetoric + hints) and context
    assert short[-1]["content"] == full[-1]["content"] + CAP  # escalation kept, cap added


def test_cap_is_at_control_length():
    # CONTROL messages had a median of ~60 words; LENGTH_MATCHED targets ~200.
    assert 40 <= CFG.short_persuasive_max_words <= 80
    assert CFG.short_persuasive_max_words < CFG.length_matched_words_followup / 2


def test_counted_as_persuasion():
    saved = CFG.persuasion_mode
    try:
        CFG.persuasion_mode = PersuasionMode.SHORT_PERSUASIVE
        assert CFG.persuasion_enabled
    finally:
        CFG.persuasion_mode = saved


def test_cli_accepts_the_mode():
    import subprocess
    out = subprocess.run([sys.executable, os.path.join(ROOT, "run_eval.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert "short_persuasive" in out.stdout
