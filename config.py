import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv
import litellm

load_dotenv()

litellm.set_verbose = False
litellm.suppress_debug_info = True


class PersuasionMode(str, Enum):
    CONTROL          = "control"           # neutral facts only, no persuasion
    RHETORIC_ONLY    = "rhetoric_only"     # persuasive framing, no trap hints, no escalation
    HINTS_ONLY       = "hints_only"        # neutral base + adversarial trap hints only
    ESCALATION_ONLY  = "escalation_only"   # neutral base + per-turn escalation templates only
    FULL             = "full"              # persuasive framing + trap hints + escalation

@dataclass
class ModelConfig:
    name: str
    provider: str
    api_key_env: str
    provider_routing: list = field(default_factory=list)

MODELS = {
    "gemma-4-31b": ModelConfig(
        "openrouter/google/gemma-4-31B-it", "openrouter", "OPENROUTER_API_KEY"
    ),
    "llama-3.3-70b": ModelConfig(
        "openrouter/meta-llama/llama-3.3-70b-instruct", "openrouter", "OPENROUTER_API_KEY"
    ),
    "gpt-4.1-mini": ModelConfig(
        "openrouter/openai/gpt-4.1-mini", "openrouter", "OPENROUTER_API_KEY"
    ),
    "qwen3-32b": ModelConfig(
        "openrouter/qwen/qwen3-32b", "openrouter", "OPENROUTER_API_KEY",
        provider_routing=["Together"]
    ),
    "deepseek-v3.2": ModelConfig(
        "openrouter/deepseek/deepseek-v3.2", "openrouter", "OPENROUTER_API_KEY"
    ),
}

@dataclass
class DatasetEntry:
    path: str
    label: str
    enabled: bool = True


def _default_datasets() -> List[DatasetEntry]:
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    return [
        DatasetEntry(path=str(f), label=f.stem, enabled=True)
        for f in sorted(data_dir.glob("*.json"))
    ]

@dataclass
class PipelineConfig:
    dev_models: List[str]   = field(default_factory=lambda: list(MODELS.keys()))
    admin_models: List[str] = field(default_factory=lambda: list(MODELS.keys()))
    max_turns: int          = 10
    judge_model: str        = "gpt-4.1-mini"
    persuasion_temperature: float = 0.35
    control_temperature: float = 0.1
    results_dir: str        = "results"
    persuasion_mode: PersuasionMode = PersuasionMode.FULL
    max_workers: int = 1

    @property
    def persuasion_enabled(self) -> bool:
        return self.persuasion_mode != PersuasionMode.CONTROL

    datasets: List[DatasetEntry] = field(default_factory=_default_datasets)

CFG = PipelineConfig()
