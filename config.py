import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    name: str
    provider: str
    api_key_env: str

MODELS = {
    "gpt-4o":         ModelConfig("gpt-4o",               "openai",    "OPENAI_API_KEY"),
    "claude-sonnet":  ModelConfig("claude-sonnet-4-5",    "anthropic", "ANTHROPIC_API_KEY"),
    "claude-opus":    ModelConfig("claude-opus-4-5",      "anthropic", "ANTHROPIC_API_KEY"),
    "gemini-2.0-pro": ModelConfig("gemini/gemini-2.0-pro","google",    "GOOGLE_API_KEY"),
    "llama-70b":      ModelConfig("ollama/llama3.3:70b",  "ollama",    ""),
}

@dataclass
class DatasetEntry:
    path: str
    label: str
    enabled: bool = True

@dataclass
class PipelineConfig:
    dev_models: List[str]   = field(default_factory=lambda: list(MODELS.keys()))
    admin_models: List[str] = field(default_factory=lambda: list(MODELS.keys()))
    max_turns: int          = 5
    judge_model: str        = "gpt-4o"
    temperature: float      = 0.7
    results_dir: str        = "results"

    datasets: List[DatasetEntry] = field(default_factory=lambda: [
        DatasetEntry("data/dataset_a_algo_tradeoffs.json", label="a_algo"),
        DatasetEntry("data/dataset_a_security_speed.json", label="a_security"),
        DatasetEntry("data/dataset_a_readability.json",    label="a_readability"),
        DatasetEntry("data/dataset_a_precision.json",      label="a_precision"),
        DatasetEntry("data/dataset_b_redundancy.json",     label="b_redundancy"),
        DatasetEntry("data/dataset_b_security.json",       label="b_security"),
        DatasetEntry("data/dataset_b_objective.json",      label="b_objective"),
        DatasetEntry("data/dataset_b_no_trap.json",        label="b_no_trap"),
    ])

CFG = PipelineConfig()
