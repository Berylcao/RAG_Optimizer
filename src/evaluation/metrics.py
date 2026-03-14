import logging
import re
from typing import Iterable, List, Optional

import numpy as np

from .base import BaseMetric
from src.pipelines.dspy_modules import (
    dspy_ready,
    judge_citation_quality_with_dspy,
    judge_groundedness_with_dspy,
)

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+")
CITATION_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "if",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
}

_SIMILARITY_MODEL_LABEL: Optional[str] = None
_SIMILARITY_MODEL = None


def _get_similarity_model():
    global _SIMILARITY_MODEL_LABEL, _SIMILARITY_MODEL
    if _SIMILARITY_MODEL is not None:
        return _SIMILARITY_MODEL
    try:
        from src.utils.local_model_registry import resolve_local_model_path
        from sentence_transformers import SentenceTransformer

        for label in ("e5-base", "bge-small"):
            try:
                path = resolve_local_model_path("embedding_models", label, require_exists=True)
                _SIMILARITY_MODEL = SentenceTransformer(str(path))
                _SIMILARITY_MODEL_LABEL = label
                logger.info("Loaded embedding model '%s' for semantic metrics.", label)
                return _SIMILARITY_MODEL
            except FileNotFoundError:
                continue
    except Exception:
        logger.debug("Semantic similarity model unavailable; falling back to token overlap.")
    return None


def _semantic_cosine(text_a: str, text_b: str) -> float:
    model = _get_similarity_model()
    if model is None:
        return -1.0
    embeddings = model.encode([text_a, text_b], normalize_embeddings=True, show_progress_bar=False)
    return float(np.dot(embeddings[0], embeddings[1]))


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _content_tokens(text: str) -> List[str]:
    return [token for token in _tokens(text) if token not in STOPWORDS]


def _token_set(text_or_chunks: str | Iterable[str]) -> set[str]:
    if isinstance(text_or_chunks, str):
        return set(_content_tokens(text_or_chunks))
    return set(_content_tokens(" ".join(text_or_chunks)))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _token_f1(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    precision = _safe_divide(overlap, len(a_tokens))
    recall = _safe_divide(overlap, len(b_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class ContextRecall(BaseMetric):
    @property
    def name(self) -> str:
        return "context_recall"

    def compute(self, retrieved_context: List[str], reference_context: str) -> float:
        if not reference_context:
            return 1.0
        if not retrieved_context:
            return 0.0
        reference_tokens = _token_set(reference_context)
        retrieved_tokens = _token_set(retrieved_context)
        return _safe_divide(len(reference_tokens & retrieved_tokens), len(reference_tokens))


class AnswerSimilarity(BaseMetric):
    """Blends semantic cosine similarity (0.6) with token F1 (0.4).

    Falls back to pure token F1 when no local embedding model is available.
    """

    @property
    def name(self) -> str:
        return "answer_similarity"

    def compute(self, generated_answer: str, reference_answer: str) -> float:
        if not reference_answer:
            return 0.0
        token_score = _token_f1(_token_set(generated_answer), _token_set(reference_answer))
        semantic_score = _semantic_cosine(generated_answer, reference_answer)
        if semantic_score < 0:
            return token_score
        return 0.6 * max(0.0, semantic_score) + 0.4 * token_score


class Faithfulness(BaseMetric):
    @property
    def name(self) -> str:
        return "faithfulness"

    def compute(self, generated_answer: str, retrieved_context: List[str]) -> float:
        if not generated_answer:
            return 0.0
        answer_tokens = _token_set(generated_answer)
        context_tokens = _token_set(retrieved_context)
        if not answer_tokens:
            return 0.0
        return _safe_divide(len(answer_tokens & context_tokens), len(answer_tokens))


class DocIDHitRate(BaseMetric):
    @property
    def name(self) -> str:
        return "doc_id_hit_rate"

    def compute(self, retrieved_doc_ids: List[str], reference_doc_ids: List[str]) -> float:
        if not reference_doc_ids:
            return 0.0
        reference_set = set(reference_doc_ids)
        hits = len([doc_id for doc_id in dict.fromkeys(retrieved_doc_ids) if doc_id in reference_set])
        return _safe_divide(hits, len(reference_set))


class RetrievalCoverageProxy(DocIDHitRate):
    @property
    def name(self) -> str:
        return "retrieval_coverage_proxy"


class Groundedness(BaseMetric):
    @property
    def name(self) -> str:
        return "groundedness"

    def compute(
        self,
        generated_answer: str,
        retrieved_context: List[str],
        query: str | None = None,
        strict_mode: bool = False,
    ) -> float:
        if not generated_answer or not retrieved_context:
            return 0.0
        if query:
            if not dspy_ready("TEACHER"):
                raise RuntimeError("Teacher judge is unavailable — TEACHER model not configured.")
            return judge_groundedness_with_dspy(
                query=query,
                contexts=retrieved_context,
                answer=generated_answer,
                model_type="TEACHER",
            )
        answer_tokens = _token_set(generated_answer)
        context_tokens = _token_set(retrieved_context)
        if not answer_tokens:
            return 0.0
        overlap = len(answer_tokens & context_tokens)
        return _safe_divide(overlap, len(answer_tokens))


class CitationQuality(BaseMetric):
    @property
    def name(self) -> str:
        return "citation_quality"

    def compute(
        self,
        generated_answer: str,
        retrieved_doc_ids: List[str],
        retrieved_context: List[str] | None = None,
        query: str | None = None,
        strict_mode: bool = False,
    ) -> float:
        if query and retrieved_context:
            if not dspy_ready("TEACHER"):
                raise RuntimeError("Teacher citation judge is unavailable — TEACHER model not configured.")
            return judge_citation_quality_with_dspy(
                query=query,
                contexts=retrieved_context,
                answer=generated_answer,
                retrieved_doc_ids=retrieved_doc_ids,
                model_type="TEACHER",
            )
        citations = CITATION_RE.findall(generated_answer or "")
        if not citations:
            return 0.0
        retrieved_set = set(retrieved_doc_ids)
        valid = len([citation for citation in citations if citation in retrieved_set])
        return _safe_divide(valid, len(citations))


class AnswerRelevance(BaseMetric):
    @property
    def name(self) -> str:
        return "answer_relevance"

    def compute(self, query: str, generated_answer: str) -> float:
        query_tokens = _token_set(query)
        answer_tokens = _token_set(generated_answer)
        if not query_tokens or not answer_tokens:
            return 0.0
        return _safe_divide(len(query_tokens & answer_tokens), len(query_tokens))