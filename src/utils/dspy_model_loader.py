import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

try:
    import dspy
except ImportError:  # pragma: no cover - optional dependency
    dspy = None

load_dotenv(override=True)


def dspy_is_installed() -> bool:
    return dspy is not None


def _normalize_api_base(api_base: Optional[str]) -> Optional[str]:
    if not api_base:
        return api_base
    normalized = api_base.rstrip("/")
    if ("localhost" in normalized or "127.0.0.1" in normalized) and not normalized.endswith("/v1"):
        normalized = normalized + "/v1"
    return normalized


def _resolve_litellm_model_name(model_name: str, api_base: Optional[str]) -> str:
    if "/" in model_name:
        return model_name

    normalized_base = (api_base or "").lower()
    if "dashscope" in normalized_base:
        return f"dashscope/{model_name}"
    return f"openai/{model_name}"


def get_model_config(
    model_type: str,
    raise_on_missing: bool = False,
    override: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    load_dotenv(override=True)
    prefix = model_type.upper()
    override = override or {}
    model_name = override.get("model_name", os.getenv(f"{prefix}_MODEL_NAME"))
    api_key = override.get("api_key", os.getenv(f"{prefix}_API_KEY"))
    api_base = override.get("api_base", os.getenv(f"{prefix}_API_BASE"))

    if not model_name:
        if raise_on_missing:
            raise ValueError(f"Please configure {prefix}_MODEL_NAME in .env")
        return None

    api_base = _normalize_api_base(api_base)
    if "localhost" in (api_base or "") or "127.0.0.1" in (api_base or ""):
        if not api_key:
            api_key = "sk-dummy-key-for-ollama"

    return {
        "role": prefix,
        "raw_model_name": model_name,
        "model_name": _resolve_litellm_model_name(model_name, api_base),
        "api_key": api_key,
        "api_base": api_base,
    }

_LM_CACHE = {}
def get_dspy_lm(model_type: str, raise_on_missing: bool = False, use_cache: bool = True):
    if dspy is None:
        if raise_on_missing:
            raise RuntimeError("DSPy is not installed.")
        return None

    config = get_model_config(model_type, raise_on_missing=raise_on_missing)
    if config is None:
        return None

    lm = dspy.LM(
        model=config["model_name"],
        api_key=config["api_key"],
        api_base=config["api_base"],
        model_type="chat",
        timeout=120, #
        max_tokens=1024,
        max_retries=2,
        temperature=0.0,
        connection_pool_size=50
    )
    if use_cache:
        _LM_CACHE[model_type] = lm
    return lm


def configure_dspy_lm(model_type: str = "TEACHER") -> bool:
    if dspy is None:
        return False

    lm = get_dspy_lm(model_type, raise_on_missing=False)
    if lm is None:
        return False
    dspy.settings.configure(lm=lm)
    return True


def configure_default_dspy_lm() -> bool:
    return configure_dspy_lm("TEACHER")