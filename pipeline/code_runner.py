import subprocess
import tempfile
import textwrap
import os
import ast
import re
from dataclasses import dataclass
from enum import Enum

TIMEOUT_SECONDS = 10

class TestStatus(str, Enum):
    PASSED        = "PASSED"
    FAILED        = "FAILED"
    SYNTAX_ERROR  = "SYNTAX_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIMEOUT       = "TIMEOUT"
    EMPTY_CODE    = "EMPTY_CODE"

@dataclass
class TestResult:
    status: TestStatus
    passed: bool
    output: str
    error: str
    assertions_total: int
    assertions_passed: int
    execution_time_ms: float

@dataclass
class SurvivalResult:
    survival_rate: float
    lines_matched: int
    lines_total_dev: int
    approved: bool


def validate_syntax(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def extract_imports(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "socket",
    "requests", "urllib", "http", "ftplib", "smtplib",
    "ctypes", "importlib", "pickle", "shelve",
    "multiprocessing", "threading",
}

def check_dangerous_imports(code: str) -> tuple[bool, list[str]]:
    found = [imp for imp in extract_imports(code) if imp.split(".")[0] in BLOCKED_IMPORTS]
    return len(found) == 0, found


def count_assertions(unit_tests: str) -> int:
    try:
        tree = ast.parse(unit_tests)
        return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))
    except SyntaxError:
        return unit_tests.count("assert ")


def build_instrumented_script(code: str, unit_tests: str) -> str:
    lines = unit_tests.strip().splitlines()
    wrapped_tests = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("assert"):
            wrapped_tests.append(textwrap.dedent(f"""
try:
    {stripped}
    __passed__ += 1
except AssertionError as e:
    print(f"ASSERTION_FAIL: {stripped[:80]!r} | {{e}}")
except Exception as e:
    print(f"RUNTIME_FAIL: {stripped[:80]!r} | {{e}}")
"""))
        elif stripped:
            wrapped_tests.append(stripped)

    return textwrap.dedent(f"""
import time as __time__
__start__ = __time__.time()

{code}

__passed__ = 0
__total__  = {count_assertions(unit_tests)}
{"".join(wrapped_tests)}

__elapsed__ = (__time__.time() - __start__) * 1000
print(f"ASSERTIONS_PASSED: {{__passed__}}/{{__total__}}")
print(f"ELAPSED_MS: {{__elapsed__:.2f}}")
if __passed__ == __total__:
    print("ALL_TESTS_PASSED")
""")


SAFE_ENV = {
    "PATH":                    "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING":        "utf-8",
}

def run_unit_tests(code: str, unit_tests: str) -> TestResult:
    if not code or not code.strip():
        return TestResult(TestStatus.EMPTY_CODE, False, "", "No code provided.",
                          count_assertions(unit_tests), 0, 0.0)

    syntax_ok, syntax_err = validate_syntax(code)
    if not syntax_ok:
        return TestResult(TestStatus.SYNTAX_ERROR, False, "", syntax_err,
                          count_assertions(unit_tests), 0, 0.0)

    is_safe, blocked = check_dangerous_imports(code)
    if not is_safe:
        return TestResult(TestStatus.RUNTIME_ERROR, False, "",
                          f"Blocked dangerous imports: {blocked}",
                          count_assertions(unit_tests), 0, 0.0)

    script = build_instrumented_script(code, unit_tests)
    total_assertions = count_assertions(unit_tests)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="eval_run_") as f:
        f.write(script)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            ["python3", tmp_path],
            capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS, env=SAFE_ENV,
        )
        stdout, stderr = proc.stdout, proc.stderr

        passed_count, elapsed_ms = 0, 0.0
        match = re.search(r"ASSERTIONS_PASSED:\s*(\d+)/(\d+)", stdout)
        if match:
            passed_count = int(match.group(1))
        match_t = re.search(r"ELAPSED_MS:\s*([\d.]+)", stdout)
        if match_t:
            elapsed_ms = float(match_t.group(1))

        all_passed = "ALL_TESTS_PASSED" in stdout and proc.returncode == 0

        if all_passed:
            status = TestStatus.PASSED
        elif "ASSERTION_FAIL" in stdout:
            status = TestStatus.FAILED
        elif stderr or proc.returncode != 0:
            status = TestStatus.RUNTIME_ERROR
        else:
            status = TestStatus.FAILED

        return TestResult(status, all_passed, stdout, stderr,
                          total_assertions, passed_count, elapsed_ms)

    except subprocess.TimeoutExpired:
        return TestResult(TestStatus.TIMEOUT, False, "",
                          f"Timed out after {TIMEOUT_SECONDS}s",
                          total_assertions, 0, TIMEOUT_SECONDS * 1000.0)
    except Exception as e:
        return TestResult(TestStatus.RUNTIME_ERROR, False, "", str(e),
                          total_assertions, 0, 0.0)
    finally:
        os.unlink(tmp_path)


def compute_code_survival_rate(dev_commit: str, merged_code: str | None) -> SurvivalResult:
    if not merged_code:
        return SurvivalResult(0.0, 0, 0, False)

    def normalize(code: str) -> list[str]:
        return [l.strip() for l in code.splitlines() if l.strip()]

    dev_lines    = normalize(dev_commit)
    merged_lines = set(normalize(merged_code))

    if not dev_lines:
        return SurvivalResult(1.0, 0, 0, True)

    matched = sum(1 for line in dev_lines if line in merged_lines)
    return SurvivalResult(round(matched / len(dev_lines), 4), matched, len(dev_lines), True)


def detect_hallucinated_imports(code: str) -> tuple[list[str], list[str]]:
    import sys, pkgutil
    stdlib_modules = set(m.name for m in pkgutil.iter_modules()) | set(sys.stdlib_module_names)
    KNOWN_PACKAGES = {
        "numpy", "pandas", "scipy", "sklearn", "matplotlib",
        "torch", "tensorflow", "keras", "flask", "fastapi",
        "requests", "pydantic", "sqlalchemy", "pytest",
        "litellm", "openai", "anthropic", "google",
        "PIL", "cv2", "yaml", "toml", "dotenv",
    }
    all_known  = stdlib_modules | KNOWN_PACKAGES
    imported   = extract_imports(code)
    hallucinated, unverifiable = [], []
    for mod in imported:
        root = mod.split(".")[0]
        if root not in all_known:
            if re.search(r'\d', root) or len(root) < 2:
                hallucinated.append(root)
            else:
                unverifiable.append(root)
    return hallucinated, unverifiable
