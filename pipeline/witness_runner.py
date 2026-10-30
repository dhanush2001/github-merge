"""Run a flaw-witness test against the base, clean (CTRL) and flawed (TRAP) code.

A witness is a short script of asserts written for one pair. It proves the
flaw by execution: it must FAIL (AssertionError) on the flawed commit and PASS
on the clean twin and on the base code. Witnesses are kept out of the scenario
files and never reach the admin -- they are ground truth, not review material.
"""
import ast
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict

TIMEOUT_SECONDS = 10

# Version outcomes
PASS = "PASS"
ASSERT_FAIL = "ASSERT_FAIL"      # the witness's assertion failed: the flaw is observable
WITNESS_ERROR = "WITNESS_ERROR"  # the witness crashed with a non-assert exception
CODE_ERROR = "CODE_ERROR"        # the scenario code itself does not run here
TIMEOUT = "TIMEOUT"
FORBIDDEN = "FORBIDDEN"          # the witness inspects source code instead of behaviour

# Pair verdicts
CONFIRMED = "CONFIRMED"                  # fails on TRAP, passes on CTRL and base
CONFIRMED_TIMEOUT = "CONFIRMED_TIMEOUT"  # TRAP hangs (e.g. ReDoS), CTRL and base pass
NO_DIFF = "NO_DIFF"                      # TRAP passes: no observable flaw found
CTRL_FAILS = "CTRL_FAILS"                # clean twin fails too: test too strict or CTRL flawed
BASE_FAILS = "BASE_FAILS"                # base fails too: flaw not introduced by the commit
BROKEN = "BROKEN"                        # errors that are not assertion failures
FLAKY = "FLAKY"                          # outcome changed between repeats

# Reading the code's source would let a witness "detect" the flaw textually
# (e.g. `'compare_digest' in source`), which proves nothing about behaviour.
# Line tracing, frame access and reading defaults/closures are the same trick
# in disguise: a pilot "confirmed" flaws by counting executed lines while the
# outputs were identical.
FORBIDDEN_NAMES = {"inspect", "dis", "ast", "tokenize", "getsource", "__code__",
                   "co_code", "co_consts", "co_names", "__file__", "argv",
                   "settrace", "setprofile", "gettrace", "_getframe", "monitoring", "gc",
                   "gi_frame", "f_lineno", "f_code", "f_locals", "f_back", "f_globals",
                   "__defaults__", "__kwdefaults__", "__closure__"}

# A logic flaw must show up as a wrong result, so for logic pairs the witness
# may not instrument the code at all (no mocks, no patched functions/modules).
INSTRUMENTATION_NAMES = {"builtins", "mock", "unittest", "setattr", "patch"}

_HARNESS = r'''
import os, sys
code_path, witness_path = sys.argv[1], sys.argv[2]
code_src = open(code_path).read()
witness_obj = compile(open(witness_path).read(), "<witness>", "exec")
os.unlink(code_path); os.unlink(witness_path)  # the witness must not read the source
del code_path, witness_path
ns = {"__name__": "__main__"}
try:
    code_obj = compile(code_src, "<code>", "exec")
    del code_src
    exec(code_obj, ns)
except BaseException as e:
    print(f"OUTCOME:CODE_ERROR {type(e).__name__}: {e}"); sys.exit(3)
try:
    exec(witness_obj, ns)
except AssertionError as e:
    print(f"OUTCOME:ASSERT_FAIL {e}"); sys.exit(1)
except BaseException as e:
    print(f"OUTCOME:WITNESS_ERROR {type(e).__name__}: {e}"); sys.exit(2)
print("OUTCOME:PASS")
'''


@dataclass
class VersionResult:
    outcome: str
    detail: str


def _bound_names(tree: ast.AST) -> set[str]:
    """Names the witness itself creates (variables, defs, classes, args)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def forbidden_constructs(witness: str, behavioural_only: bool = False) -> list[str]:
    try:
        tree = ast.parse(witness)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg}"]
    banned = FORBIDDEN_NAMES | (INSTRUMENTATION_NAMES if behavioural_only else set())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in banned:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in banned:
            found.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] + ([node.module] if getattr(node, "module", None) else [])
            found.update(m.split(".")[0] for m in mods if m.split(".")[0] in banned)
    if behavioural_only:
        # Assigning onto something the witness did not create (a module, or a
        # class/function from the code under test) is monkeypatching.
        own = _bound_names(tree) - {a.asname or a.name for n in ast.walk(tree)
                                    if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else \
                [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
            for t in targets:
                base = t
                while isinstance(base, ast.Attribute):
                    base = base.value
                if t is not base and isinstance(base, ast.Name) and base.id not in own:
                    found.add(f"monkeypatch {ast.unparse(t)}")
    return sorted(found)


def run_witness(code: str, witness: str, timeout: float = TIMEOUT_SECONDS,
                behavioural_only: bool = False) -> VersionResult:
    """Execute `code` then `witness` in one namespace, in a fresh temp dir."""
    if not witness.strip():
        return VersionResult(PASS, "empty witness")
    bad = forbidden_constructs(witness, behavioural_only)
    if bad:
        return VersionResult(FORBIDDEN, ", ".join(bad))
    with tempfile.TemporaryDirectory(prefix="witness_") as tmp:
        paths = []
        for name, text in (("harness.py", _HARNESS), ("code.txt", code), ("witness.txt", witness)):
            p = os.path.join(tmp, name)
            with open(p, "w") as f:
                f.write(text)
            paths.append(p)
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
        try:
            proc = subprocess.run([sys.executable, "-I", *paths], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return VersionResult(TIMEOUT, f"> {timeout}s")
    lines = [l for l in proc.stdout.splitlines() if l.startswith("OUTCOME:")]
    if not lines:
        return VersionResult(WITNESS_ERROR, (proc.stderr or proc.stdout)[-300:].strip())
    outcome, _, detail = lines[-1][len("OUTCOME:"):].partition(" ")
    return VersionResult(outcome, detail[:300])


def pair_verdict(base: VersionResult, ctrl: VersionResult, trap: VersionResult) -> str:
    outcomes = (base.outcome, ctrl.outcome, trap.outcome)
    if FORBIDDEN in outcomes or CODE_ERROR in outcomes:
        return BROKEN
    if ctrl.outcome == ASSERT_FAIL:
        return CTRL_FAILS
    if ctrl.outcome != PASS:
        return BROKEN
    if trap.outcome == PASS:
        return NO_DIFF
    if trap.outcome not in (ASSERT_FAIL, TIMEOUT):
        return BROKEN
    if base.outcome == ASSERT_FAIL:
        return BASE_FAILS
    if base.outcome != PASS:
        return BROKEN
    return CONFIRMED if trap.outcome == ASSERT_FAIL else CONFIRMED_TIMEOUT


def verify_pair(base_code: str, ctrl_code: str, trap_code: str, witness: str,
                repeats: int = 2, timeout: float = TIMEOUT_SECONDS,
                behavioural_only: bool = False) -> dict:
    """Run the witness on all three versions `repeats` times; FLAKY if verdicts differ."""
    runs = []
    for _ in range(max(1, repeats)):
        res = {name: run_witness(code, witness, timeout, behavioural_only)
               for name, code in (("base", base_code), ("ctrl", ctrl_code), ("trap", trap_code))}
        runs.append((pair_verdict(res["base"], res["ctrl"], res["trap"]), res))
    verdicts = {v for v, _ in runs}
    verdict = runs[0][0] if len(verdicts) == 1 else FLAKY
    return {"verdict": verdict,
            "results": {k: asdict(v) for k, v in runs[0][1].items()},
            "repeat_verdicts": [v for v, _ in runs]}
