"""Checkpoint management for fault-tolerant evaluation runs.

Saves progress after every task so a run interrupted by a token-limit error
(or any other failure) can be resumed with --resume.

Checkpoint file: results/checkpoint_<run_id>.json
"""

import json
import os
import glob
import tempfile
import threading
from typing import Any, Dict, List, Optional


class TokenLimitReached(Exception):
    """Raised when an LLM call fails because the context window was exceeded.

    Carries ``partial_state`` — everything needed to resume the negotiation
    from the exact turn and agent that hit the limit.
    """

    def __init__(self, scenario_id: str, dev_model: str, admin_model: str, partial_state: Dict[str, Any]):
        self.scenario_id = scenario_id
        self.dev_model = dev_model
        self.admin_model = admin_model
        self.partial_state = partial_state
        turn = partial_state.get("turn_num", "?")
        who = partial_state.get("who", "?")
        super().__init__(
            f"Token limit reached for {scenario_id} (dev={dev_model}, admin={admin_model})"
            f" at turn {turn} in {who} call"
        )


class RateLimitReached(Exception):
    """Raised when an LLM call fails due to a provider rate limit (HTTP 429).

    Carries ``partial_state`` identical in structure to ``TokenLimitReached``
    so the run can be resumed with ``--resume`` after the limit resets.
    """

    def __init__(self, scenario_id: str, dev_model: str, admin_model: str, partial_state: Dict[str, Any]):
        self.scenario_id = scenario_id
        self.dev_model = dev_model
        self.admin_model = admin_model
        self.partial_state = partial_state
        turn = partial_state.get("turn_num", "?")
        who = partial_state.get("who", "?")
        super().__init__(
            f"Rate limit hit for {scenario_id} (dev={dev_model}, admin={admin_model})"
            f" at turn {turn} in {who} call"
        )


def _task_key(scenario_id: str, dev_model: str, admin_model: str) -> str:
    return "|".join([scenario_id, dev_model, admin_model])


class CheckpointManager:
    """Thread-safe checkpoint manager.

    Creates / loads a JSON file at ``results/checkpoint_<run_id>.json`` and
    writes it to disk after every state change so interrupted runs can be
    resumed.

    Stored entry structure
    ----------------------
    Each task key maps to one of::

        {"status": "completed",   "result":     <result_dict>}
        {"status": "token_limit", "checkpoint": <partial_state>}
        {"status": "error",       "error":      "<message>"}
    """

    VERSION = "1"

    def __init__(self, results_dir: str, run_id: str):
        os.makedirs(results_dir, exist_ok=True)
        self.path = os.path.join(results_dir, f"checkpoint_{run_id}.json")
        self.run_id = run_id
        self._data: Dict = {"version": self.VERSION, "run_id": run_id, "tasks": {}}
        self._lock = threading.Lock()

    @classmethod
    def from_file(cls, path: str) -> "CheckpointManager":
        """Load an existing checkpoint file, auto-recovering if it was truncated."""
        with open(path, "rb") as f:
            raw = f.read()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = cls._recover_truncated(raw, path)

        basename = os.path.basename(path)
        run_id = basename[len("checkpoint_"):-len(".json")]

        mgr = cls.__new__(cls)
        mgr.path = os.path.abspath(path)
        mgr.run_id = run_id
        mgr._data = data
        mgr._lock = threading.Lock()
        mgr._save_locked()
        return mgr

    @staticmethod
    def _recover_truncated(raw: bytes, path: str) -> dict:
        """Find the last complete task entry in a truncated checkpoint file."""
        import shutil
        bak = path + ".bak"
        shutil.copy(path, bak)
        print(f"[WARN] Checkpoint truncated — recovering. Backup saved to {bak}")

        end = len(raw)
        while True:
            idx = raw.rfind(b"},", 0, end)
            if idx == -1:
                break
            candidate = raw[:idx + 2] + b"\n}\n}"
            try:
                data = json.loads(candidate)
                print(f"[WARN] Recovered {len(data.get('tasks', {}))} tasks from truncated checkpoint.")
                return data
            except json.JSONDecodeError:
                end = idx
                continue

        raise RuntimeError(f"Could not recover checkpoint: {path}")

    def _save_locked(self):
        """Write checkpoint to disk atomically. Caller must already hold self._lock."""
        dir_ = os.path.dirname(self.path)
        with tempfile.NamedTemporaryFile(mode='w', dir=dir_, delete=False, suffix=".tmp") as tmp:
            json.dump(self._data, tmp, indent=2, default=str)
            tmp_path = tmp.name
        os.replace(tmp_path, self.path)

    def is_completed(self, scenario_id: str, dev_model: str, admin_model: str) -> bool:
        key = _task_key(scenario_id, dev_model, admin_model)
        return self._data.get("tasks", {}).get(key, {}).get("status") == "completed"

    def all_completed_results(self) -> List[Dict]:
        """Return all stored result dicts for successfully completed tasks."""
        return [
            entry["result"]
            for entry in self._data["tasks"].values()
            if entry.get("status") == "completed"
        ]

    def summary(self) -> Dict[str, int]:
        """Count tasks per status (completed / token_limit / error)."""
        counts: Dict[str, int] = {}
        for entry in self._data["tasks"].values():
            status = entry.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def mark_completed(self, scenario_id: str, dev_model: str, admin_model: str, result_dict: dict):
        key = _task_key(scenario_id, dev_model, admin_model)
        with self._lock:
            self._data["tasks"][key] = {"status": "completed", "result": result_dict}
            self._save_locked()

    def mark_error(self, scenario_id: str, dev_model: str, admin_model: str, error_str: str):
        key = _task_key(scenario_id, dev_model, admin_model)
        with self._lock:
            self._data["tasks"][key] = {"status": "error", "error": error_str}
            self._save_locked()


def find_latest_checkpoint(results_dir: str) -> Optional[str]:
    """Return the path of the most recently written checkpoint file, or None."""
    pattern = os.path.join(results_dir, "checkpoint_*.json")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None
