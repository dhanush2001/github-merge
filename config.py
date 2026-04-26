import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import litellm

load_dotenv()

litellm.set_verbose = False
litellm.suppress_debug_info = True

@dataclass
class ModelConfig:
    name: str
    provider: str
    api_key_env: str

MODELS = {
    "openrouter-claude-opus": ModelConfig("openrouter/anthropic/claude-opus-4-5", "openrouter", "OPENROUTER_API_KEY"),
    "openrouter-claude-sonnet": ModelConfig("openrouter/anthropic/claude-sonnet-4", "openrouter", "OPENROUTER_API_KEY"),
    "azure-gpt4o":    ModelConfig("azure/gpt-4o-deployment", "azure", "AZURE_API_KEY"),
}

@dataclass
class DatasetEntry:
    path: str
    label: str
    enabled: bool = True


def _default_datasets() -> List[DatasetEntry]:
    """Load every JSON dataset under data/ and enable them by default."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []

    entries: List[DatasetEntry] = []
    for dataset_file in sorted(data_dir.glob("*.json")):
        entries.append(
            DatasetEntry(
                path=str(dataset_file),
                label=dataset_file.stem,
                enabled=True,
            )
        )
    return entries

@dataclass
class PipelineConfig:
    dev_models: List[str]   = field(default_factory=lambda: list(MODELS.keys()))
    admin_models: List[str] = field(default_factory=lambda: list(MODELS.keys()))
    max_turns: int          = 10
    judge_model: str        = "azure-gpt4o"
    persuasion_temperature: float = 0.35
    control_temperature: float = 0.1
    results_dir: str        = "results"
    persuasion_enabled: bool = True

    datasets: List[DatasetEntry] = field(default_factory=_default_datasets)

CFG = PipelineConfig()
