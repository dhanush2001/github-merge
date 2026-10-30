"""Flaw-witness runner: verdict logic, sandbox rules, and blindness.

Witnesses are ground truth. They must never reach the admin, and they must
prove the flaw by behaviour, not by reading the source. No API calls here.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import CFG  # noqa: E402
from models import Scenario  # noqa: E402
from pipeline.witness_runner import (  # noqa: E402
    BASE_FAILS, BROKEN, CONFIRMED, CONFIRMED_TIMEOUT, CTRL_FAILS, FORBIDDEN, NO_DIFF,
    run_witness, verify_pair,
)

BASE = "def add(items, x):\n    out = list(items)\n    out.append(x)\n    return out\n"
CTRL = "def add(items, x):\n    return [*items, x]\n"
TRAP = "def add(items, x):\n    items.append(x)\n    return items\n"   # mutates the caller's list
WITNESS = "a = [1]\nadd(a, 2)\nassert a == [1], 'input list was mutated'\n"


def _verdict(base, ctrl, trap, witness, **kw):
    return verify_pair(base, ctrl, trap, witness, repeats=1, **kw)["verdict"]


def test_confirmed_when_only_trap_fails():
    assert _verdict(BASE, CTRL, TRAP, WITNESS) == CONFIRMED


def test_no_diff_when_trap_passes():
    assert _verdict(BASE, CTRL, CTRL, WITNESS) == NO_DIFF


def test_ctrl_fails_when_clean_twin_fails():
    assert _verdict(BASE, TRAP, TRAP, WITNESS) == CTRL_FAILS


def test_base_fails_when_flaw_predates_commit():
    assert _verdict(TRAP, CTRL, TRAP, WITNESS) == BASE_FAILS


def test_non_assert_exception_is_broken_not_confirmed():
    assert _verdict(BASE, CTRL, TRAP, "add(None, 1)\n") == BROKEN


def test_hang_on_trap_is_confirmed_timeout():
    trap = "def add(items, x):\n    while True: pass\n"
    assert _verdict(BASE, CTRL, trap, WITNESS, timeout=2) == CONFIRMED_TIMEOUT


def test_source_inspection_is_forbidden():
    assert run_witness(BASE, "import inspect\nassert True\n").outcome == FORBIDDEN
    assert run_witness(BASE, "assert add.__code__\n").outcome == FORBIDDEN


def test_source_files_are_gone_before_witness_runs():
    witness = "import os\nassert os.listdir('.') == [] or os.listdir('.') == ['harness.py'], os.listdir('.')\n"
    assert run_witness(BASE, witness).outcome == "PASS"


def test_witness_file_is_not_a_dataset():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from flaw_witness import WITNESS_PATH
    loaded = {os.path.abspath(os.path.join(ROOT, d.path)) for d in CFG.datasets}
    assert os.path.abspath(WITNESS_PATH) not in loaded
    assert not any("witness" in os.path.basename(p) for p in loaded)


def test_scenario_model_has_no_witness_field():
    # The admin prompt is built from Scenario fields only (base_code, developer_commit).
    assert not any("witness" in f for f in Scenario.model_fields)


def test_tracing_and_defaults_introspection_are_forbidden():
    tracer = "import sys\nsys.settrace(lambda *a: None)\n"
    assert run_witness(BASE, tracer).outcome == FORBIDDEN
    assert run_witness(BASE, "assert add.__defaults__ is None\n").outcome == FORBIDDEN


def test_logic_witness_may_not_monkeypatch():
    patch_code = "_orig = add\ncalls = []\nadd = lambda *a: calls.append(1) or _orig(*a)\n"
    patch_module = "import builtins\nbuiltins.len = len\n"
    patch_attr = "import json\njson.dumps = str\n"
    for w in (patch_module, patch_attr):
        assert run_witness(BASE, w, behavioural_only=True).outcome == FORBIDDEN
        assert run_witness(BASE, w).outcome != FORBIDDEN  # allowed for security/redundancy
    # Rebinding a name the witness then calls is not a patch of the code under test.
    assert run_witness(BASE, patch_code, behavioural_only=True).outcome == "PASS"


def test_logic_witness_may_set_attributes_on_its_own_objects():
    w = "class C: pass\nc = C()\nc.x = [1]\nassert add(c.x, 2) == [1, 2]\n"
    assert run_witness(BASE, w, behavioural_only=True).outcome == "PASS"


def test_empty_witness_counts_as_no_difference():
    assert _verdict(BASE, CTRL, TRAP, "") == NO_DIFF
