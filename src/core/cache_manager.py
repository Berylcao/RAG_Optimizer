"""
Disk-level cache for chunking and indexing artefacts.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


def _stable_hash(*blobs: str) -> str:
    """Return a short hex digest that is deterministic across runs."""
    h = hashlib.sha256()
    for b in blobs:
        h.update(b.encode("utf-8"))
    return h.hexdigest()[:16]


class CacheManager:
    """Simple JSON-file cache for expensive pipeline artefacts."""

    def __init__(self, cache_dir: Path | str | None = None, enabled: bool = True):
        self._dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._enabled = enabled
        self._chunks_dir = self._dir / "chunks"
        if self._enabled:
            self._chunks_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def make_chunk_key(
        documents: List[Dict[str, Any]],
        chunking_config: Dict[str, Any],
    ) -> str:
        doc_blob = json.dumps(
            [{"doc_id": d.get("doc_id", ""), "content": d.get("content", "")} for d in documents],
            sort_keys=True,
            ensure_ascii=False,
        )
        cfg_blob = json.dumps(chunking_config, sort_keys=True, ensure_ascii=False)
        return _stable_hash(doc_blob, cfg_blob)

    def get_chunks(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if not self._enabled:
            return None
        path = self._chunks_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info("Disk cache HIT for chunks key=%s (%d chunks).", key, len(data))
            return data
        except Exception:
            logger.warning("Corrupt cache entry %s; ignoring.", path)
            path.unlink(missing_ok=True)
            return None

    def put_chunks(self, key: str, chunks: List[Dict[str, Any]]) -> None:
        if not self._enabled:
            return
        path = self._chunks_dir / f"{key}.json"
        try:
            path.write_text(
                json.dumps(chunks, ensure_ascii=False, indent=None),
                encoding="utf-8",
            )
            logger.info("Disk cache STORE chunks key=%s (%d chunks).", key, len(chunks))
        except Exception:
            logger.warning("Failed to write cache entry %s.", path, exc_info=True)

    def clear(self) -> None:
        """Remove all cached artefacts."""
        if not self._dir.exists():
            return
        count = 0
        for f in self._chunks_dir.glob("*.json"):
            f.unlink(missing_ok=True)
            count += 1
        logger.info("Cleared %d cache entries from %s.", count, self._chunks_dir)
