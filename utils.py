import json
import os
import re
import threading
from collections import Counter
from typing import Dict, Optional, Tuple

import litellm

from config import MODELS, azure_api_base, api_version_for_model


_PARSE_STATS_LOCK = threading.Lock()
_PARSE_STATS: Counter = Counter()


def record_parse_result(model_name: str, ok: bool) -> None:
    key = f"{model_name}:{'ok' if ok else 'fail'}"
    with _PARSE_STATS_LOCK:
        _PARSE_STATS[key] += 1


def get_parse_stats() -> Dict[str, int]:
    with _PARSE_STATS_LOCK:
        return dict(_PARSE_STATS)


def require_model(model_key: str):
    """Validate model key and API key presence, return ModelConfig."""
    if model_key not in MODELS:
        valid = ", ".join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown model key '{model_key}'. Valid keys: {valid}")
    cfg = MODELS[model_key]
    if cfg.api_key_env and not os.getenv(cfg.api_key_env):
        raise ValueError(f"Missing required environment variable: {cfg.api_key_env}")
    return cfg


def provider_kwargs(model_cfg) -> Dict[str, object]:
    """Build LiteLLM provider kwargs for the given ModelConfig."""
    if model_cfg is None:
        return {}
    provider = getattr(model_cfg, "provider", "")
    kwargs: Dict[str, object] = {}
    if provider == "azure":
        kwargs = {
            "api_key": os.getenv(getattr(model_cfg, "api_key_env", "AZURE_API_KEY")),
            "api_base": azure_api_base(),
            "api_version": api_version_for_model(model_cfg),
        }
    elif provider == "openrouter":
        kwargs = {
            "api_key": os.getenv(getattr(model_cfg, "api_key_env", "OPENROUTER_API_KEY")),
            "api_base": os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        }
    else:
        return {}

    effort = getattr(model_cfg, "reasoning_effort", None)
    if effort and "gpt-5" in getattr(model_cfg, "name", "").lower():
        kwargs["reasoning_effort"] = effort
    return kwargs


def normalize_message_content(content) -> str:
    """Normalize LLM message.content to a string.

    LiteLLM/OpenRouter sometimes returns content as:
      - str (normal case)
      - list of parts (multipart content, e.g. [{"type": "text", "text": "..."}])
      - dict (already-parsed JSON when response_format={"type":"json_object"})
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if text:
            return str(text)
        return json.dumps(content)
    return str(content)


def count_text_tokens(model_name: str, text: str) -> int:
    if not text:
        return 0
    try:
        return int(
            litellm.token_counter(
                model=model_name,
                messages=[{"role": "assistant", "content": text}],
            )
        )
    except Exception:
        return max(1, len(text) // 4)


def extract_usage(response, model_name: str, text: str) -> Tuple[int, int]:
    """Return (input_tokens, output_tokens) from response.usage, falling back to counting."""
    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if output_tokens == 0:
            output_tokens = count_text_tokens(model_name, text)
        return input_tokens, output_tokens
    return 0, count_text_tokens(model_name, text)


def extract_cached_input_tokens(response) -> int:
    """Return the number of cached prompt tokens reported by Azure prompt caching, else 0."""
    usage = getattr(response, "usage", None)
    if not usage:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
    if details is None:
        return 0
    cached = getattr(details, "cached_tokens", None)
    if cached is None and isinstance(details, dict):
        cached = details.get("cached_tokens")
    return int(cached or 0)


_FENCE_JSON_RE = re.compile(r"```(?:json|JSON)?\s*(\{[\s\S]*?\})\s*```")


def extract_json_block(text: str) -> Dict:
    """Robustly extract a JSON dict from text using multiple fallback strategies."""
    text = (text or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    # Markdown fence anywhere in the text (Qwen/Llama often wrap JSON in ```json ... ```)
    fence_match = _FENCE_JSON_RE.search(text)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            try:
                parsed = json.loads(stripped)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


_DECISION_KW_RE = re.compile(r"\b(APPROVE|REJECT|CLARIFY)\b", re.IGNORECASE)


def keyword_decision_fallback(raw_text: str) -> Optional[Dict]:
    """Last-resort: scan first ~500 chars for a decision keyword.

    Used when JSON parsing entirely fails (e.g. Qwen returning prose).
    Returns a synthetic payload dict, or None if no keyword is found.
    """
    snippet = (raw_text or "")[:500]
    m = _DECISION_KW_RE.search(snippet)
    if not m:
        return None
    return {
        "decision": m.group(1).upper(),
        "feedback": (raw_text or "")[:1000],
        "confidence": 2,
    }
