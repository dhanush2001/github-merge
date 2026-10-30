"""Both agents sample identically in every condition.

The May/July runs used temperature 0.1 in the control condition and 0.35 under
persuasion, for the developer and the admin, so part of the persuasion effect
could have been sampling randomness. Conditions may now differ ONLY in the
developer's prompt: these tests record the real API requests (no API calls are
made) and require every other setting to match across conditions, and the
developer and admin to use the same temperature.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import agents.developer_agent as developer_agent  # noqa: E402
from config import CFG, MODELS, PersuasionMode  # noqa: E402
from test_admin_settings import _admin_request, _scenario  # noqa: E402


class _Recorded(Exception):
    pass


def _dev_request(monkeypatch, model_key, mode):
    seen = {}

    def fake_completion(**kwargs):
        seen.update(kwargs)
        raise _Recorded

    monkeypatch.setattr(developer_agent.litellm, "completion", fake_completion)
    monkeypatch.setattr(CFG, "persuasion_mode", mode)
    with pytest.raises(_Recorded):
        developer_agent.call_developer(_scenario(), model_key, [], "", turn=1)
    return seen


@pytest.mark.parametrize("model_key", sorted(MODELS))
def test_developer_settings_identical_in_every_condition(monkeypatch, model_key):
    reqs = {m.value: _dev_request(monkeypatch, model_key, m) for m in PersuasionMode}
    settings = {m: {k: v for k, v in r.items() if k != "messages"} for m, r in reqs.items()}
    baseline = settings[PersuasionMode.CONTROL.value]
    for mode, s in settings.items():
        assert s == baseline, f"{model_key}: developer settings differ in mode={mode}"


@pytest.mark.parametrize("model_key", sorted(MODELS))
def test_developer_and_admin_use_the_same_temperature(monkeypatch, model_key):
    for mode in PersuasionMode:
        dev = _dev_request(monkeypatch, model_key, mode)
        admin = _admin_request(monkeypatch, model_key, mode)
        assert dev["temperature"] == admin["temperature"], f"{model_key} {mode.value}"


def test_configured_temperature_is_used(monkeypatch):
    # A non-GPT-5 model uses CFG.temperature (GPT-5 models are forced to 1.0).
    assert _dev_request(monkeypatch, "llama-3.3-70b", PersuasionMode.CONTROL)["temperature"] == CFG.temperature


@pytest.mark.parametrize("model_key", sorted(k for k in MODELS if "gpt" in k))
def test_gpt_models_get_seed_42(monkeypatch, model_key):
    # GPT models honour `seed` (best-effort determinism); keep it fixed at 42.
    assert CFG.seed == 42
    for mode in (PersuasionMode.CONTROL, PersuasionMode.FULL):
        assert _dev_request(monkeypatch, model_key, mode)["seed"] == 42
        assert _admin_request(monkeypatch, model_key, mode)["seed"] == 42
