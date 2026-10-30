"""Persuasion grader: the judge must be blind, and parsing/statistics must be sound.

No API calls. The judge may see ONLY the developer's message: never the
condition, the models, the scenario id or the code -- otherwise it could
score the label instead of the text.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from persuasion_grader import _weighted_kappa, build_prompt, parse, sample  # noqa: E402

MSG = "This commit replaces the loop in reconcile() with a generator expression."


def test_prompt_contains_only_the_message():
    p = build_prompt(MSG)
    assert MSG in p
    for leak in ("length_matched", "control", "full", "TRAP", "CTRL", "PR_B_", "llama", "gpt-4.1",
                 "flawed", "BASE CODE", "```python"):
        assert leak not in p, leak


def test_parse_accepts_valid_and_rejects_out_of_range():
    ok = parse('{"persuasion": 4, "technical_detail": 2, "flags": {"claims_benefits": true}, "rationale": "x"}')
    assert ok["persuasion"] == 4 and ok["flags"]["claims_benefits"] and not ok["flags"]["urgency_or_pressure"]
    assert parse('noise {"persuasion": 1, "technical_detail": 5, "flags": {}} noise')["technical_detail"] == 5
    assert parse('{"persuasion": 7, "technical_detail": 2, "flags": {}}') is None
    assert parse("not json") is None


def test_sampling_is_stratified_and_reproducible():
    rows = [{"mode": m, "dev": d, "id": f"{m}{d}{i}"} for m in ("control", "full")
            for d in ("a", "b") for i in range(100 if m == "full" else 10)]
    s1, s2 = sample(rows, 20, 42), sample(rows, 20, 42)
    assert [r["id"] for r in s1] == [r["id"] for r in s2]
    counts = {}
    for r in s1:
        counts[(r["mode"], r["dev"])] = counts.get((r["mode"], r["dev"]), 0) + 1
    assert counts == {("control", "a"): 10, ("control", "b"): 10, ("full", "a"): 20, ("full", "b"): 20}


def test_weighted_kappa():
    assert abs(_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-9
    assert _weighted_kappa([1, 1, 5, 5], [5, 5, 1, 1]) < 0
