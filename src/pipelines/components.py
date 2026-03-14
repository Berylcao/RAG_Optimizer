import logging
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from openai import OpenAI

from .base import (
    BaseChunker,
    BaseGenerator,
    BaseQueryRefiner,
    BaseReranker,
    BaseRetriever,
    Chunk,
    RetrievalResult,
)
from .dspy_modules import dspy_ready, generate_answer_with_dspy, rewrite_query_with_dspy
from src.utils.dspy_model_loader import get_model_config
from src.utils.local_model_registry import encode_with_local_model, get_cross_encoder

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "a", "an", "and", "are", "be", "before", "can", "do", "does", "for", "how",
    "if", "in", "is", "it", "of", "on", "or", "should", "the", "to", "what",
    "when", "which", "with", "without",
}
ACRONYM_EXPANSIONS = {
    "crm": "customer relationship management",
    "kyc": "know your customer",
    "rag": "retrieval augmented generation",
    "sme": "small medium enterprise",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> List[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def build_text(doc: Dict[str, Any], metadata_enrichment: bool) -> str:
    base_text = normalize_text(doc["text"])
    if not metadata_enrichment:
        return base_text
    title = normalize_text(doc.get("title", ""))
    source = normalize_text(doc.get("source", ""))
    prefix = " ".join(part for part in [title, source] if part)
    return normalize_text(f"{prefix}. {base_text}") if prefix else base_text


def split_sentences(text: str) -> List[str]:
    sentences = [normalize_text(sentence) for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    return sentences or [normalize_text(text)]


def semantic_merge_sentences(
    sentences: List[str],
    chunk_size: int,
    embeddings: np.ndarray | None = None,
    similarity_threshold: float = 0.5,
) -> List[str]:
    """Merge consecutive sentences into chunks using embedding cosine similarity.

    When *embeddings* is provided (one vector per sentence), adjacent sentences
    are merged while (a) cosine similarity exceeds *similarity_threshold* and
    (b) total token count stays within *chunk_size*.  Falls back to token-overlap
    heuristic when embeddings are unavailable.
    """
    if not sentences:
        return []
    use_embeddings = embeddings is not None and len(embeddings) == len(sentences)
    merged: List[str] = []
    current = [sentences[0]]
    current_tokens = len(sentences[0].split())
    previous_terms = set(content_tokens(sentences[0]))

    for idx, sentence in enumerate(sentences[1:], start=1):
        sentence_tokens = len(sentence.split())
        if use_embeddings:
            sim = float(np.dot(embeddings[idx - 1], embeddings[idx]))
            should_merge = sim >= similarity_threshold
        else:
            sentence_terms = set(content_tokens(sentence))
            should_merge = len(previous_terms & sentence_terms) > 0

        if current and current_tokens + sentence_tokens <= chunk_size and should_merge:
            current.append(sentence)
            current_tokens += sentence_tokens
            if not use_embeddings:
                previous_terms |= set(content_tokens(sentence))
        else:
            merged.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
            if not use_embeddings:
                previous_terms = set(content_tokens(sentence))
    if current:
        merged.append(" ".join(current))
    return merged


def chunk_by_tokens(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_by_sentences(text: str, chunk_size: int, overlap: int) -> List[str]:
    sentences = split_sentences(text)
    grouped: List[str] = []
    current: List[str] = []
    current_tokens = 0
    overlap_tokens = max(0, overlap)

    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if current and current_tokens + sentence_tokens > chunk_size:
            grouped.append(" ".join(current))
            if overlap_tokens > 0:
                retained: List[str] = []
                retained_tokens = 0
                for item in reversed(current):
                    retained.insert(0, item)
                    retained_tokens += len(item.split())
                    if retained_tokens >= overlap_tokens:
                        break
                current = retained
                current_tokens = retained_tokens
            else:
                current = []
                current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        grouped.append(" ".join(current))
    return grouped


def lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = content_tokens(query)
    text_tokens = content_tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    query_counts = Counter(query_tokens)
    text_counts = Counter(text_tokens)
    overlap = sum(min(query_counts[token], text_counts[token]) for token in query_counts)
    return overlap / max(1, sum(query_counts.values()))


def dense_like_score(query: str, text: str) -> float:
    query_terms = set(content_tokens(query))
    text_terms = set(content_tokens(text))
    if not query_terms or not text_terms:
        return 0.0
    intersection = len(query_terms & text_terms)
    union = len(query_terms | text_terms)
    coverage = intersection / max(1, len(query_terms))
    jaccard = intersection / max(1, union)
    return 0.7 * coverage + 0.3 * jaccard


def rank_chunks(
    chunks: Sequence[Chunk],
    query: str,
    scoring_fn,
    top_k: int,
    retriever_name: str,
) -> RetrievalResult:
    scored: List[Tuple[float, Chunk]] = []
    for chunk in chunks:
        score = scoring_fn(query, chunk.content)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:top_k]
    return RetrievalResult(
        chunks=[chunk for _, chunk in top],
        scores=[score for score, _ in top],
        query=query,
        effective_query=query,
        retriever_name=retriever_name,
        metadata={},
    )


class TokenChunker(BaseChunker):
    def __init__(self, chunk_size: int, overlap: int, metadata_enrichment: bool = False):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.metadata_enrichment = metadata_enrichment

    def chunk(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            text = build_text(doc, self.metadata_enrichment)
            parts = chunk_by_tokens(text, self.chunk_size, self.overlap)
            if not parts:
                parts = [text]
            for index, part in enumerate(parts):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc['doc_id']}-tok-{index}",
                        content=part,
                        metadata={"title": doc.get("title"), "source": doc.get("source")},
                        doc_id=doc["doc_id"],
                    )
                )
        logger.info("Created %s token chunks.", len(chunks))
        return chunks


class SentenceChunker(BaseChunker):
    def __init__(self, chunk_size: int, overlap: int, metadata_enrichment: bool = False):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.metadata_enrichment = metadata_enrichment

    def chunk(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            text = build_text(doc, self.metadata_enrichment)
            parts = chunk_by_sentences(text, self.chunk_size, self.overlap)
            if not parts:
                parts = [text]
            for index, part in enumerate(parts):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc['doc_id']}-sent-{index}",
                        content=part,
                        metadata={"title": doc.get("title"), "source": doc.get("source")},
                        doc_id=doc["doc_id"],
                    )
                )
        logger.info("Created %s sentence chunks.", len(chunks))
        return chunks


class SemanticChunker(BaseChunker):
    def __init__(self, chunk_size: int, overlap: int, metadata_enrichment: bool = False, embed_model: str = "bge-small"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.metadata_enrichment = metadata_enrichment
        self.embed_model = embed_model

    def _encode_sentences(self, sentences: List[str]) -> np.ndarray | None:
        if not sentences:
            return None
        try:
            return encode_with_local_model(self.embed_model, sentences)
        except Exception:
            logger.debug("Embedding model unavailable for semantic chunking; using token-overlap fallback.")
            return None

    def chunk(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            text = build_text(doc, self.metadata_enrichment)
            sentences = split_sentences(text)
            embeddings = self._encode_sentences(sentences)
            merged = semantic_merge_sentences(sentences, self.chunk_size, embeddings=embeddings)
            if not merged:
                merged = [text]
            for index, part in enumerate(merged):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc['doc_id']}-sem-{index}",
                        content=part,
                        metadata={
                            "title": doc.get("title"),
                            "source": doc.get("source"),
                            "embedding_model": self.embed_model,
                        },
                        doc_id=doc["doc_id"],
                    )
                )
        logger.info("Created %s semantic chunks.", len(chunks))
        return chunks


class BM25Retriever(BaseRetriever):
    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._bm25 = None

    def index(self, chunks: List[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        tokenized_corpus = [tokenize(chunk.content) for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Indexed %s chunks for BM25 retrieval.", len(self._chunks))

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not self._chunks or self._bm25 is None:
            return RetrievalResult(chunks=[], scores=[], query=query, effective_query=query, retriever_name="bm25")
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        order = np.argsort(scores)[::-1][:top_k]
        return RetrievalResult(
            chunks=[self._chunks[i] for i in order],
            scores=[float(scores[i]) for i in order],
            query=query,
            effective_query=query,
            retriever_name="bm25",
            metadata={},
        )


class DenseRetriever(BaseRetriever):
    def __init__(self, embed_model: str):
        self.embed_model = embed_model
        self._chunks: List[Chunk] = []
        self._embeddings = None

    def index(self, chunks: List[Chunk]) -> None:
        self._chunks = list(chunks)
        if self._chunks:
            self._embeddings = encode_with_local_model(
                self.embed_model,
                [chunk.content for chunk in self._chunks],
            )
        logger.info("Indexed %s chunks for dense retrieval (%s).", len(self._chunks), self.embed_model)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not self._chunks:
            return RetrievalResult(chunks=[], scores=[], query=query, effective_query=query, retriever_name="dense")
        query_embedding = encode_with_local_model(self.embed_model, [query])[0]
        scores = np.matmul(self._embeddings, query_embedding)
        order = np.argsort(scores)[::-1][:top_k]
        return RetrievalResult(
            chunks=[self._chunks[index] for index in order],
            scores=[float(scores[index]) for index in order],
            query=query,
            effective_query=query,
            retriever_name="dense",
            metadata={"embedding_model": self.embed_model},
        )


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridRetriever(BaseRetriever):
    def __init__(self, embed_model: str, alpha: float = 0.5):
        self.embed_model = embed_model
        self.alpha = alpha
        self._chunks: List[Chunk] = []
        self._embeddings = None
        self._bm25 = None

    def index(self, chunks: List[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        if self._chunks:
            self._embeddings = encode_with_local_model(
                self.embed_model,
                [chunk.content for chunk in self._chunks],
            )
            tokenized_corpus = [tokenize(chunk.content) for chunk in self._chunks]
            self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Indexed %s chunks for hybrid retrieval.", len(self._chunks))

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not self._chunks:
            return RetrievalResult(chunks=[], scores=[], query=query, effective_query=query, retriever_name="hybrid")
        query_embedding = encode_with_local_model(self.embed_model, [query])[0]
        dense_raw = np.matmul(self._embeddings, query_embedding)
        sparse_raw = self._bm25.get_scores(tokenize(query))
        dense_norm = _min_max_normalize(dense_raw)
        sparse_norm = _min_max_normalize(sparse_raw)
        hybrid_scores = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm
        order = np.argsort(hybrid_scores)[::-1][:top_k]
        return RetrievalResult(
            chunks=[self._chunks[i] for i in order],
            scores=[float(hybrid_scores[i]) for i in order],
            query=query,
            effective_query=query,
            retriever_name="hybrid",
            metadata={"embedding_model": self.embed_model, "alpha": self.alpha},
        )


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def rerank(self, query: str, result: RetrievalResult, top_n: int = 5) -> RetrievalResult:
        if not result.chunks:
            return result
        model = get_cross_encoder(self.model_name)
        pairs = [[query, chunk.content] for chunk in result.chunks]
        rerank_scores = model.predict(pairs)
        scored = [
            (float(score), chunk)
            for score, chunk in zip(rerank_scores, result.chunks)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:top_n]
        return RetrievalResult(
            chunks=[chunk for _, chunk in top],
            scores=[score for score, _ in top],
            query=result.query,
            effective_query=result.effective_query,
            retriever_name=f"{result.retriever_name}+rerank",
            metadata={"reranker_model": self.model_name},
        )


class HeuristicQueryRefiner(BaseQueryRefiner):
    def __init__(self, rewrite_enabled: bool = True, decompose_enabled: bool = False):
        self.rewrite_enabled = rewrite_enabled
        self.decompose_enabled = decompose_enabled

    def _rewrite(self, query: str) -> str:
        expanded_terms: List[str] = []
        base_tokens = content_tokens(query)
        for token in base_tokens:
            if token in ACRONYM_EXPANSIONS:
                expanded_terms.append(ACRONYM_EXPANSIONS[token])
        normalized = normalize_text(query)
        if expanded_terms:
            normalized = f"{normalized} {' '.join(expanded_terms)}"
        return normalized

    def _decompose(self, query: str) -> str:
        parts = [part.strip() for part in re.split(r"\band\b|\bor\b|,", query, flags=re.IGNORECASE) if part.strip()]
        if len(parts) <= 1:
            return query
        return f"{query} {' '.join(parts)}"

    def refine(self, query: str) -> str:
        refined = normalize_text(query)
        if self.rewrite_enabled:
            refined = self._rewrite(refined)
        if self.decompose_enabled:
            refined = self._decompose(refined)
        return refined


class DSPyQueryRefiner(BaseQueryRefiner):
    def __init__(self, rewrite_enabled: bool = True, decompose_enabled: bool = False, strict_mode: bool = False):
        self.rewrite_enabled = rewrite_enabled
        self.decompose_enabled = decompose_enabled
        self.strict_mode = strict_mode
        self._heuristic = HeuristicQueryRefiner(
            rewrite_enabled=False,
            decompose_enabled=decompose_enabled,
        )

    def refine(self, query: str) -> str:
        if not self.rewrite_enabled:
            return self._heuristic.refine(query)
        if not dspy_ready("TEACHER"):
            raise RuntimeError("DSPy query rewriting is unavailable — TEACHER model not configured.")
        try:
            rewritten = rewrite_query_with_dspy(query, model_type="TEACHER")
            if self.decompose_enabled:
                rewritten = self._heuristic._decompose(rewritten)
            return normalize_text(rewritten)
        except Exception as exc:
            raise RuntimeError("DSPy query rewriting failed.") from exc


class StudentChatGenerator(BaseGenerator):
    def __init__(
        self,
        llm_name: str,
        answer_style: str = "concise",
        temperature: float = 0.0,
        strict_mode: bool = False,
        model_config_override: Dict[str, Any] | None = None,
    ):
        self.llm_name = llm_name
        self.answer_style = answer_style
        self.temperature = temperature
        self.strict_mode = strict_mode
        self.model_config_override = model_config_override or {}

    def _build_context_block(self, retrieval_result: RetrievalResult) -> str:
        parts: List[str] = []
        for chunk in retrieval_result.chunks:
            parts.append(f"[{chunk.doc_id}] {chunk.content}")
        return "\n".join(parts)

    def generate(self, query: str, retrieval_result: RetrievalResult) -> str:
        if not retrieval_result.chunks:
            return "Insufficient grounded context to answer the query."
        config = get_model_config(
            "STUDENT",
            raise_on_missing=True,
            override=self.model_config_override,
        )

        system_prompt = (
            "You are a grounded RAG answer generator. "
            "Answer only from the provided context. "
            "If the context is insufficient, say so explicitly."
        )
        citation_instruction = (
            "Cite supporting document ids inline like [D001]."
            if self.answer_style == "citation_first"
            else "Do not invent citations."
        )
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Context:\n{self._build_context_block(retrieval_result)}\n\n"
            f"Instructions:\n"
            f"- {citation_instruction}\n"
            f"- Keep the answer concise.\n"
        )

        try:
            client = OpenAI(
                api_key=config["api_key"],
                base_url=config["api_base"],
                timeout=120.0,
            )
            response = client.chat.completions.create(
                model=config["raw_model_name"],
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                raise RuntimeError("Student model returned empty response.")
            return normalize_text(answer)
        except Exception as exc:
            raise RuntimeError("Student model generation failed.") from exc


class DSPyGenerator(BaseGenerator):
    def __init__(self, llm_name: str = "gpt-class", answer_style: str = "concise", temperature: float = 0.0, strict_mode: bool = False):
        self.llm_name = llm_name
        self.answer_style = answer_style
        self.temperature = temperature
        self.strict_mode = strict_mode

    def generate(self, query: str, retrieval_result: RetrievalResult) -> str:
        if not dspy_ready("STUDENT"):
            raise RuntimeError("DSPy student generation is unavailable — STUDENT model not configured.")
        try:
            answer = generate_answer_with_dspy(query, retrieval_result.contexts, model_type="STUDENT")
            if self.answer_style == "citation_first" and retrieval_result.doc_ids:
                missing = [did for did in retrieval_result.doc_ids if f"[{did}]" not in answer]
                if missing:
                    answer = f"{answer} [{'] ['.join(missing)}]"
            return normalize_text(answer)
        except Exception as exc:
            raise RuntimeError("DSPy generation failed.") from exc