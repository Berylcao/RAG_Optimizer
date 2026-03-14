from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = ROOT_DIR / "configs" / "model_paths.yaml"


@lru_cache(maxsize=1)
def load_model_registry() -> Dict:
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_models_base_path() -> Path:
    config = load_model_registry()
    base_path = config.get("models_base_path", "./models")
    return (ROOT_DIR / base_path).resolve()


def get_model_entry(model_group: str, model_label: str) -> Dict[str, str]:
    config = load_model_registry()
    group = config.get(model_group, {})
    if model_label not in group:
        raise KeyError(f"Unknown {model_group} label: {model_label}")
    entry = group[model_label]
    if not isinstance(entry, dict) or "local_dir" not in entry:
        raise ValueError(f"Invalid model registry entry for {model_group}.{model_label}")
    return entry


def resolve_local_model_path(model_group: str, model_label: str, require_exists: bool = True) -> Path:
    entry = get_model_entry(model_group, model_label)
    path = get_models_base_path() / entry["local_dir"]
    if require_exists and not path.exists():
        raise FileNotFoundError(
            f"Local model not found for {model_group}.{model_label}: {path}. "
            "Run `python -m src.utils.model_download_cli` before using this configuration."
        )
    return path


def iter_download_targets(
    *,
    include_disabled: bool = False,
    only_labels: Optional[set[str]] = None,
) -> Iterable[Tuple[str, str, Dict[str, str], Path]]:
    config = load_model_registry()
    base_path = get_models_base_path()
    for model_group in ("embedding_models", "reranker_models"):
        for label, entry in config.get(model_group, {}).items():
            if only_labels and label not in only_labels:
                continue
            if not include_disabled and entry.get("download_enabled", True) is False:
                continue
            yield model_group, label, entry, base_path / entry["local_dir"]


@lru_cache(maxsize=4)
def get_sentence_transformer(model_label: str):
    from sentence_transformers import SentenceTransformer

    model_path = resolve_local_model_path("embedding_models", model_label)
    return SentenceTransformer(str(model_path))


@lru_cache(maxsize=4)
def get_cross_encoder(model_label: str):
    from sentence_transformers import CrossEncoder

    model_path = resolve_local_model_path("reranker_models", model_label)
    return CrossEncoder(str(model_path))


def encode_with_local_model(model_label: str, texts: List[str]):
    model = get_sentence_transformer(model_label)
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
