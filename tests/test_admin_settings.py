"""The admin's request must not depend on the persuasion condition.

Experiments 1 and 2 ran the admin at temperature 0.1 in the control condition
and 0.35 under persuasion, so part of the persuasion effect could have been the
reviewer sampling more randomly. Only the developer may vary by condition: for
the same PR and the same developer message, the admin's entire API request
(prompt, temperature, seed, every other setting) must be identical.

No API calls are made: litellm.completion is replaced by a recorder.
"""
import glob
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import agents.admin_agent as admin_agent  # noqa: E402
from config import CFG, MODELS, PersuasionMode  # noqa: E402
from models import Scenario  # noqa: E402


class _Recorded(Exception):
    pass


def _scenario():
    path = sorted(glob.glob(os.path.join(ROOT, "data", "dataset_b_*.json")))[0]
    return Scenario(**json.load(open(path))[0])


def _admin_request(monkeypatch, model_key, mode):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        raise _Recorded

    monkeypatch.setattr(admin_agent.litellm, "completion", fake_completion)
    monkeypatch.setattr(CFG, "persuasion_mode", mode)
    with pytest.raises(_Recorded):
        admin_agent.call_admin(_scenario(), model_key, "Please merge.", turn=1)
    return seen


@pytest.mark.parametrize("model_key", sorted(MODELS))
def test_admin_request_identical_in_every_condition(monkeypatch, model_key):
    requests = {m.value: _admin_request(monkeypatch, model_key, m) for m in PersuasionMode}
    baseline = requests[PersuasionMode.CONTROL.value]
    for mode, req in requests.items():
        assert req == baseline, f"{model_key}: admin request differs in mode={mode}"


def test_admin_temperature_is_the_configured_one(monkeypatch):
    # A non-GPT-5 model uses CFG.temperature (GPT-5 is forced to 1.0).
    req = _admin_request(monkeypatch, "llama-3.3-70b", PersuasionMode.FULL)
    assert req["temperature"] == CFG.temperature
