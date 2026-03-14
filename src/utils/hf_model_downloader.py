"""
Model download utilities for local embedding and reranker models.
"""
import logging
import os
from typing import Optional

from huggingface_hub import snapshot_download

from .local_model_registry import get_models_base_path, iter_download_targets

logger = logging.getLogger(__name__)

DEFAULT_IGNORE_PATTERNS = [
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "rust_model.ot",
    "*.tflite",
]

DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "*.bin",
    "tokenizer.*",
    "vocab.*",
    "merges.txt",
    "sentence_*.json",
    "modules.json",
    "special_tokens_map.json",
    "config.json",
    "1_Pooling/*",
    "2_Dense/*",
]


def _resolve_patterns(model_group: str) -> tuple[list[str], list[str]]:
    allow_patterns = list(DEFAULT_ALLOW_PATTERNS)
    if model_group == "reranker_models":
        allow_patterns.extend(["*.py"])
    return allow_patterns, list(DEFAULT_IGNORE_PATTERNS)


def download_all_models(
    *,
    use_mirror: bool = False,
    only_labels: Optional[list[str]] = None,
    hf_token: Optional[str] = None,
) -> None:
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("Using HF mirror endpoint: https://hf-mirror.com")

    token = hf_token or os.getenv("HF_TOKEN")
    models_base_path = get_models_base_path()
    models_base_path.mkdir(parents=True, exist_ok=True)
    logger.info("Target download path: %s", models_base_path)

    only_labels_set = set(only_labels or [])
    targets = list(
        iter_download_targets(
            include_disabled=bool(only_labels_set),
            only_labels=only_labels_set or None,
        )
    )
    if not targets:
        logger.warning("No download targets matched the current selection.")
        return

    logger.info("Selected download targets:")
    for model_group, label, entry, local_dir in targets:
        logger.info("%s: %s -> %s -> %s", model_group, label, entry["repo_id"], local_dir)

    for model_group, label, entry, local_dir in targets:
        repo_id = entry["repo_id"]
        allow_patterns, ignore_patterns = _resolve_patterns(model_group)

        logger.info(
            "Downloading model: group=%s label=%s repo_id=%s local_dir=%s",
            model_group,
            label,
            repo_id,
            local_dir,
        )

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            logger.info("Completed: [%s]", label)
        except Exception as exc:
            logger.exception("Failed: [%s] %s", label, exc)
            logger.warning("Try a different endpoint, add HF_TOKEN, or retry later.")
