"""Static baseline scoring: a commit is flagged only for findings its base lacks."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from static_baselines import CONFIGS, evaluate, new_findings, summarise  # noqa: E402

PAIRS = [{"pair_id": "P1", "category": "Security_Trap", "witness": "CONFIRMED"},
         {"pair_id": "P2", "category": "Objective_Trap", "witness": "NO_DIFF"}]


def test_findings_already_in_base_do_not_count():
    found = {("P1", "base"): [("B608", 2)], ("P1", "trap"): [("B608", 2)]}
    assert not new_findings(found, "P1", "trap", 1)


def test_an_extra_occurrence_of_a_base_rule_counts():
    found = {("P1", "base"): [("B608", 2)], ("P1", "trap"): [("B608", 2), ("B608", 2)]}
    assert new_findings(found, "P1", "trap", 1) == {"B608": 1}


def test_severity_threshold():
    found = {("P1", "trap"): [("B110", 1)]}
    assert new_findings(found, "P1", "trap", 1) and not new_findings(found, "P1", "trap", 2)


def test_rates_and_witness_subset():
    findings = {"bandit": {("P1", "trap"): [("B608", 2)], ("P2", "ctrl"): [("B110", 1)]},
                "semgrep": {}}
    rows = evaluate(PAIRS, findings)
    assert len(rows) == len(PAIRS) * len(CONFIGS)
    s = {(r["subset"], r["config"], r["category"]): r for r in summarise(rows)}
    a = s[("all_pairs", "bandit_any", "All")]
    assert (a["flags_flawed"], a["flags_clean"], a["discriminates"]) == (0.5, 0.5, 0.5)
    w = s[("witness_confirmed", "bandit_any", "All")]
    assert (w["n_pairs"], w["flags_flawed"], w["flags_clean"]) == (1, 1.0, 0.0)
    assert s[("all_pairs", "bandit_medium+", "All")]["flags_clean"] == 0.0  # B110 is LOW
