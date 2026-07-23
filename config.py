import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv
import litellm
from litellm.caching.caching import Cache

load_dotenv(override=True)

litellm.set_verbose = False
litellm.suppress_debug_info = True

LITELLM_CACHE_DIR = os.getenv("LITELLM_CACHE_DIR", ".litellm_cache")
if os.getenv("LITELLM_CACHE_DISABLE", "0") != "1":
    litellm.cache = Cache(type="disk", disk_cache_dir=LITELLM_CACHE_DIR)

DEFAULT_AZURE_API_BASE = "https://dk-eval-r.services.ai.azure.com"
DEFAULT_AZURE_API_VERSION = "2025-01-01-preview"
MODEL_API_VERSION_DEFAULTS = {
    "AZURE_GPT_5_1_API_VERSION":   "2025-04-01-preview",
    "AZURE_GPT_5_MINI_API_VERSION": "2025-04-01-preview",
    "AZURE_GPT_5_4_API_VERSION":   "2025-04-01-preview",
    "AZURE_GPT_5_2_API_VERSION":   "2025-04-01-preview",
}


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
    reasoning_effort: str | None = None

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
   "qwen3.7-max": ModelConfig(
       "openrouter/qwen/qwen3.7-max",
       "openrouter",
       "OPENROUTER_API_KEY",
   ),
   "qwen3.7-plus": ModelConfig(
       "openrouter/qwen/qwen3.7-plus",
       "openrouter",
       "OPENROUTER_API_KEY",
   ),
   "qwen3.6-flash": ModelConfig(
       "openrouter/qwen/qwen3.6-flash",
       "openrouter",
       "OPENROUTER_API_KEY",
   ),
    "gpt-5-mini": ModelConfig(
        f"azure/{os.getenv('AZURE_GPT_5_MINI_DEPLOYMENT', 'gpt-5-mini')}",
        "azure",
        "AZURE_API_KEY",
        "AZURE_GPT_5_MINI_API_VERSION",
        reasoning_effort="low",
    ),
    "gpt-5.1": ModelConfig(
        f"azure/{os.getenv('AZURE_GPT_5_1_DEPLOYMENT', 'gpt-5.1')}",
        "azure",
        "AZURE_API_KEY",
        "AZURE_GPT_5_1_API_VERSION",
        reasoning_effort="low",
    ),
    "gpt-5.4": ModelConfig(
        f"azure/{os.getenv('AZURE_GPT_5_4_DEPLOYMENT', 'gpt-5.4')}",
        "azure",
        "AZURE_API_KEY",
        "AZURE_GPT_5_4_API_VERSION",
        reasoning_effort="low",
    ),
    "gpt-5.2": ModelConfig(
        f"azure/{os.getenv('AZURE_GPT_5_2_DEPLOYMENT', 'gpt-5.2')}",
        "azure",
        "AZURE_API_KEY",
        "AZURE_GPT_5_2_API_VERSION",
        reasoning_effort="low",
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
    judge_model: str        = "gpt-5.1"
    persuasion_temperature: float = 0.35
    control_temperature: float = 0.1
    seed: int               = 42
    results_dir: str        = "results"
    persuasion_mode: PersuasionMode = PersuasionMode.FULL
    max_workers: int = 16

    @property
    def persuasion_enabled(self) -> bool:
        return self.persuasion_mode != PersuasionMode.CONTROL

    datasets: List[DatasetEntry] = field(default_factory=_default_datasets)

CFG = PipelineConfig()


def completion_temperature(model_name: str, requested_temperature: float) -> float:
    if "gpt-5" in model_name.lower():
        return 1.0
    return requested_temperature


def azure_api_base() -> str:
    return os.getenv("AZURE_API_BASE", DEFAULT_AZURE_API_BASE)


def api_version_for_model(model_cfg: ModelConfig) -> str:
    env_name = getattr(model_cfg, "api_version_env", "AZURE_API_VERSION")
    configured = os.getenv(env_name)
    if configured:
        return configured
    if env_name in MODEL_API_VERSION_DEFAULTS:
        return MODEL_API_VERSION_DEFAULTS[env_name]
    return os.getenv("AZURE_API_VERSION", DEFAULT_AZURE_API_VERSION)


def validate_runtime_config(dev_models: List[str], admin_models: List[str], judge_model: str) -> None:
    all_keys = set(dev_models) | set(admin_models)
    unknown = sorted(all_keys - set(MODELS))
    errors = []
    if unknown:
        errors.append(
            f"Unknown model key(s): {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(MODELS))}"
        )
    required_envs = {MODELS[k].api_key_env for k in all_keys if MODELS[k].api_key_env}
    missing = sorted(env for env in required_envs if not os.getenv(env))
    if missing:
        errors.append(f"Missing required environment variable(s): {', '.join(missing)}")
    if errors:
        raise ValueError("Configuration error(s):\n  - " + "\n  - ".join(errors))


def format_runtime_config(dev_models: List[str], admin_models: List[str], judge_model: str) -> str:
    lines = [
        "[INFO] Runtime configuration:",
        f"  persuasion_mode : {CFG.persuasion_mode.value}",
        f"  workers         : {CFG.max_workers}",
        f"  results_dir     : {CFG.results_dir}",
        f"  dev_models      : {', '.join(dev_models)}",
        f"  admin_models    : {', '.join(admin_models)}",
        f"  judge_model     : {judge_model}",
    ]
    return "\n".join(lines)
