"""LENGTH_MATCHED: neutral tone at the persuasion condition's length.

Reviewers objected that FULL (persuasive, ~200 words) was compared with
CONTROL (neutral, ~60 words), so tone and length were confounded. This mode
keeps CONTROL's neutrality but asks for FULL's length. These tests check it
carries none of the persuasion machinery and asks for the right length.
Flawed/clean symmetry is covered by tests/test_developer_symmetry.py and equal
sampling settings by tests/test_sampling_settings.py, which run over every
PersuasionMode.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents.developer_agent import (  # noqa: E402
    DEV_BASIC_INSTRUCTION, DEV_LENGTH_MATCHED_INSTRUCTION, DEV_TRAP_HINTS, ESCALATION_TEMPLATES,
    _build_messages,
)
from config import CFG, PersuasionMode  # noqa: E402
from test_developer_symmetry import PAIRS  # noqa: E402


@pytest.fixture
def mode(request):
    saved = CFG.persuasion_mode
    CFG.persuasion_mode = request.param
    yield request.param
    CFG.persuasion_mode = saved


def _messages(turn, feedback=""):
    messages, _ = _build_messages(PAIRS[0][0], feedback, [], turn)
    return messages


@pytest.mark.parametrize("mode", [PersuasionMode.LENGTH_MATCHED], indirect=True)
@pytest.mark.parametrize("turn", [1, 2, 3, 6])
def test_no_persuasion_components(mode, turn):
    assert mode is PersuasionMode.LENGTH_MATCHED
    text = "\n".join(m["content"] for m in _messages(turn, "Why this change?" if turn > 1 else ""))
    assert DEV_BASIC_INSTRUCTION not in text
    assert DEV_TRAP_HINTS not in text
    assert "Turn strategy:" not in text
    assert not any(t in text for t in ESCALATION_TEMPLATES.values())
    assert "persuade the Administrator" not in text


@pytest.mark.parametrize("mode", [PersuasionMode.LENGTH_MATCHED], indirect=True)
def test_asks_for_full_conditions_length(mode):
    assert mode is PersuasionMode.LENGTH_MATCHED
    first, later = _messages(1), _messages(2, "Why this change?")
    assert first[0]["content"] == DEV_LENGTH_MATCHED_INSTRUCTION
    assert f"about {CFG.length_matched_words_first} words" in first[-1]["content"]
    assert f"about {CFG.length_matched_words_followup} words" in later[-1]["content"]
    assert "Do not argue for the merge." in later[-1]["content"]


def test_word_targets_are_far_above_control_length():
    # CONTROL messages had a median of ~60 words; the targets must sit at FULL's ~200.
    assert 150 <= CFG.length_matched_words_followup <= CFG.length_matched_words_first <= 250


def test_not_counted_as_persuasion():
    saved = CFG.persuasion_mode
    try:
        CFG.persuasion_mode = PersuasionMode.LENGTH_MATCHED
        assert not CFG.persuasion_enabled
    finally:
        CFG.persuasion_mode = saved


def test_cli_accepts_the_mode():
    import subprocess
    out = subprocess.run([sys.executable, os.path.join(ROOT, "run_eval.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert "length_matched" in out.stdout
