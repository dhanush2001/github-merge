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
    "openrouter-claude": ModelConfig("openrouter/anthropic/claude-sonnet-4", "openrouter", "OPENROUTER_API_KEY"),
    "azure-gpt4o":    ModelConfig("azure/gpt-4o-deployment", "azure", "AZURE_API_KEY"),
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
    max_turns: int          = 10
    judge_model: str        = "azure-gpt4o"
    temperature: float      = 0.7
    results_dir: str        = "results"

    datasets: List[DatasetEntry] = field(default_factory=lambda: [
        DatasetEntry("data/sample_MVP_dataset_A.json", label="sample_mvp_a", enabled=True),
        DatasetEntry("data/sample_MVP_dataset_B.json", label="sample_mvp_b", enabled=False),
    ])

CFG = PipelineConfig()
