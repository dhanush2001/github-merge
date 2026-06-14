import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv
import litellm

load_dotenv(override=True)

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
    api_version_env: str = "AZURE_API_VERSION"
    provider_routing: list = field(default_factory=list)

MODELS = {
    "llama-4-maverick": ModelConfig(
        "openrouter/meta-llama/llama-4-maverick",
        "openrouter",
        "OPENROUTER_API_KEY",
    ),
    "llama-4-scout": ModelConfig(
        "openrouter/meta-llama/llama-4-scout",
        "openrouter",
        "OPENROUTER_API_KEY",
    ),
    "llama-3.3-70b": ModelConfig(
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "openrouter",
        "OPENROUTER_API_KEY",
    ),
    "qwen3-max": ModelConfig(
        "openrouter/qwen/qwen3-max",
        "openrouter",
        "OPENROUTER_API_KEY",
    ),
    "qwen3-coder-plus": ModelConfig(
        "openrouter/qwen/qwen3-coder-plus",
        "openrouter",
        "OPENROUTER_API_KEY",
    ),
    "qwen3-235b-a22b": ModelConfig(
        "openrouter/qwen/qwen3-235b-a22b",
        "openrouter",
        "OPENROUTER_API_KEY",
    ),
    "gpt-5": ModelConfig(
        f"azure/{os.getenv('AZURE_GPT_5_DEPLOYMENT', 'gpt-5')}",
        "azure",
        "AZURE_API_KEY",
    ),
    "gpt-4.1": ModelConfig(
        f"azure/{os.getenv('AZURE_GPT_4_1_DEPLOYMENT', 'gpt-4.1')}",
        "azure",
        "AZURE_API_KEY",
    ),
    "gpt-5.4": ModelConfig(
        f"azure/{os.getenv('AZURE_GPT_5_4_DEPLOYMENT', 'gpt-5.4')}",
        "azure",
        "AZURE_API_KEY",
        "AZURE_GPT_5_4_API_VERSION",
    ),
    "gpt-4o": ModelConfig(
        f"azure/{os.getenv('AZURE_GPT_4O_DEPLOYMENT', 'gpt-4o-deployment')}",
        "azure",
        "AZURE_API_KEY",
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
    judge_model: str        = "gpt-4.1"
    persuasion_temperature: float = 0.35
    control_temperature: float = 0.1
    results_dir: str        = "results"
    persuasion_mode: PersuasionMode = PersuasionMode.FULL
    max_workers: int = 16

    @property
    def persuasion_enabled(self) -> bool:
        return self.persuasion_mode != PersuasionMode.CONTROL

    datasets: List[DatasetEntry] = field(default_factory=_default_datasets)

CFG = PipelineConfig()


def completion_temperature(model_name: str, requested_temperature: float) -> float:
    normalized = model_name.lower()
    if "gpt-5" in normalized and "gpt-5.1" not in normalized:
        return 1.0
    return requested_temperature
