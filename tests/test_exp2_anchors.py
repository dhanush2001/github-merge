"""Experiment 2 runs every candidate against more than one fixed anchor.

Reviewers objected that a single GPT-5.1 anchor cannot separate "this model is
a weak reviewer" from "this model is weak against GPT-5.1's style". These
tests check the second anchor is wired up and the analysis tags each trace by
its anchor and role.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from clustered_inference import exp2_role  # noqa: E402
from config import EXP2_ANCHORS, EXP2_CANDIDATES, MODELS  # noqa: E402
from utils import provider_kwargs  # noqa: E402


def test_anchors_and_candidates_are_registered_and_disjoint():
    assert EXP2_ANCHORS == ["gpt-5.1", "deepseek-v4-flash"]
    assert all(m in MODELS for m in EXP2_ANCHORS + EXP2_CANDIDATES)
    assert not set(EXP2_ANCHORS) & set(EXP2_CANDIDATES)


def test_deepseek_anchor_gets_openrouter_reasoning_setting():
    kw = provider_kwargs(MODELS["deepseek-v4-flash"])
    assert kw["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "reasoning_effort" not in kw


def test_gpt5_and_plain_openrouter_models_unchanged():
    assert provider_kwargs(MODELS["gpt-5.1"])["reasoning_effort"] == "low"
    assert "extra_body" not in provider_kwargs(MODELS["llama-4-scout"])


def test_role_tagging():
    assert exp2_role("deepseek-v4-flash", "llama-4-scout") == ("deepseek-v4-flash", "candidate_screens", "llama-4-scout")
    assert exp2_role("qwen3.7-max", "deepseek-v4-flash") == ("deepseek-v4-flash", "candidate_argues", "qwen3.7-max")
    assert exp2_role("gpt-5.1", "gpt-5.4") == ("gpt-5.1", "candidate_screens", "gpt-5.4")
    assert exp2_role("gpt-5.1", "deepseek-v4-flash") is None   # anchor vs anchor
    assert exp2_role("gpt-5.4", "llama-4-scout") is None       # candidate vs candidate
