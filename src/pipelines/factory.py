import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseChunker, BaseRetriever, BaseReranker, BaseGenerator, BaseQueryRefiner, RetrievalResult, Chunk
from .components import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    DSPyGenerator,
    DSPyQueryRefiner,
    HeuristicQueryRefiner,
    HybridRetriever,
    SemanticChunker,
    SentenceChunker,
    StudentChatGenerator,
    TokenChunker,
)
from src.core.cache_manager import CacheManager
from src.utils.dspy_model_loader import get_model_config

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.chunker: Optional[BaseChunker] = None
        self.retriever: Optional[BaseRetriever] = None
        self.reranker: Optional[BaseReranker] = None
        self.generator: Optional[BaseGenerator] = None
        self.query_refiner: Optional[BaseQueryRefiner] = None

        self.is_indexed = False
        self._chunks_cache: List[Any] = []

    def setup_components(self, config: Dict[str, Any], strict_formal_mode: bool = False):
        ck_cfg = config.get('chunking', {})
        idx_cfg = config.get('indexing', {})
        metadata_enrichment = idx_cfg.get('metadata_enrichment', False)
        if ck_cfg.get('strategy') == 'semantic':
            self.chunker = SemanticChunker(
                chunk_size=ck_cfg.get('size', 256),
                overlap=ck_cfg.get('overlap', 0),
                metadata_enrichment=metadata_enrichment,
                embed_model=idx_cfg.get('embedding_model', 'bge-small'),
            )
        elif ck_cfg.get('strategy') == 'sentence':
            self.chunker = SentenceChunker(
                chunk_size=ck_cfg.get('size', 256),
                overlap=ck_cfg.get('overlap', 0),
                metadata_enrichment=metadata_enrichment,
            )
        else:
            self.chunker = TokenChunker(
                chunk_size=ck_cfg.get('size', 256),
                overlap=ck_cfg.get('overlap', 0),
                metadata_enrichment=metadata_enrichment,
            )

        retriever_name = idx_cfg.get('retriever')
        if retriever_name == 'bm25':
            self.retriever = BM25Retriever()
        elif retriever_name == 'dense':
            self.retriever = DenseRetriever(
                embed_model=idx_cfg.get('embedding_model', 'bge-small')
            )
        else:
            self.retriever = HybridRetriever(
                embed_model=idx_cfg.get('embedding_model', 'bge-small')
            )

        rr_cfg = config.get('reranking', {})
        if rr_cfg.get('enabled'):
            self.reranker = CrossEncoderReranker(
                model_name=rr_cfg.get('model', 'cross-encoder-mini')
            )
        else:
            self.reranker = None

        qr_cfg = config.get('query_refinement', {})
        rewrite_enabled = qr_cfg.get('rewrite', False)
        decompose_enabled = qr_cfg.get('decompose', False)
        if rewrite_enabled:
            self.query_refiner = DSPyQueryRefiner(
                rewrite_enabled=True,
                decompose_enabled=decompose_enabled,
                strict_mode=strict_formal_mode,
            )
        elif decompose_enabled:
            self.query_refiner = HeuristicQueryRefiner(
                rewrite_enabled=False,
                decompose_enabled=True,
            )
        else:
            self.query_refiner = None

        gen_cfg = config.get('generation', {})
        student_config = get_model_config("STUDENT", raise_on_missing=False)
        student_label = student_config["raw_model_name"] if student_config else "student-model"
        llm_name = gen_cfg.get('llm', student_label)
        answer_style = gen_cfg.get('answer_style', 'concise')
        temperature = gen_cfg.get('temperature', 0.0)
        student_override = {
            "model_name": gen_cfg.get("student_model_name"),
            "api_base": gen_cfg.get("student_api_base"),
            "api_key": gen_cfg.get("student_api_key"),
        }
        student_override = {key: value for key, value in student_override.items() if value is not None}
        if llm_name == 'gpt-class':
            self.generator = DSPyGenerator(
                llm_name=llm_name,
                answer_style=answer_style,
                temperature=temperature,
                strict_mode=strict_formal_mode,
            )
        else:
            self.generator = StudentChatGenerator(
                llm_name=llm_name,
                answer_style=answer_style,
                temperature=temperature,
                strict_mode=strict_formal_mode,
                model_config_override=student_override,
            )

        logger.info("Pipeline components initialized.")

    def index_documents(self, documents: List[Dict]):
        if not self.chunker or not self.retriever:
            raise RuntimeError("Components not initialized. Call setup_components first.")

        self._chunks_cache = self.chunker.chunk(documents)
        self.retriever.index(self._chunks_cache)
        self.is_indexed = True
        logger.info("Indexing complete. Total chunks: %s", len(self._chunks_cache))

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not self.is_indexed:
            raise RuntimeError("Pipeline is not indexed.")

        current_query = query
        if self.query_refiner:
            current_query = self.query_refiner.refine(query)

        results = self.retriever.retrieve(current_query, top_k=top_k)
        results.query = query
        results.effective_query = current_query

        if self.reranker and results.chunks:
            results = self.reranker.rerank(current_query, results)
            results.query = query
            results.effective_query = current_query

        return results

    def generate(self, query: str, retrieval_result: RetrievalResult) -> str:
        if not self.generator:
            raise RuntimeError("Generator not initialized.")
        return self.generator.generate(query, retrieval_result)


def _index_cache_key(config: Dict[str, Any]) -> str:
    """Deterministic hash of chunking + indexing + reranking params."""
    subset = {
        "chunking": config.get("chunking", {}),
        "indexing": config.get("indexing", {}),
        "reranking": config.get("reranking", {}),
    }
    blob = json.dumps(subset, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class PipelineFactory:
    _index_cache: Dict[str, Tuple[List[Any], Any]] = {}
    _disk_cache: CacheManager = CacheManager(enabled=True)

    @classmethod
    def reset_cache(cls, disk: bool = False) -> None:
        """Clear the in-memory index cache and optionally the disk cache."""
        cls._index_cache.clear()
        if disk:
            cls._disk_cache.clear()

    @classmethod
    def configure_disk_cache(cls, enabled: bool = True, cache_dir: str | None = None) -> None:
        cls._disk_cache = CacheManager(cache_dir=cache_dir, enabled=enabled)

    @staticmethod
    def create_pipeline(config: Dict[str, Any], documents: List[Dict], strict_formal_mode: bool = False) -> RAGPipeline:
        pipeline = RAGPipeline()

        try:
            pipeline.setup_components(config, strict_formal_mode=strict_formal_mode)
            cache_key = _index_cache_key(config)

            cached = PipelineFactory._index_cache.get(cache_key)
            if cached is not None:
                chunks, _ = cached
                pipeline._chunks_cache = chunks
                pipeline.retriever.index(chunks)
                pipeline.is_indexed = True
                logger.info("Memory cache HIT key=%s (%d chunks).", cache_key, len(chunks))
                return pipeline

            disk_cache = PipelineFactory._disk_cache
            chunk_key = disk_cache.make_chunk_key(documents, config.get("chunking", {}))
            disk_data = disk_cache.get_chunks(chunk_key)
            if disk_data is not None:
                chunks = [Chunk(**d) for d in disk_data]
                pipeline._chunks_cache = chunks
                pipeline.retriever.index(chunks)
                pipeline.is_indexed = True
                PipelineFactory._index_cache[cache_key] = (chunks, None)
                logger.info("Disk cache HIT key=%s (%d chunks).", chunk_key, len(chunks))
                return pipeline

            pipeline.index_documents(documents)
            PipelineFactory._index_cache[cache_key] = (
                pipeline._chunks_cache,
                None,
            )
            disk_cache.put_chunks(
                chunk_key,
                [
                    {"chunk_id": c.chunk_id, "content": c.content,
                     "metadata": c.metadata, "doc_id": c.doc_id}
                    for c in pipeline._chunks_cache
                ],
            )
            return pipeline
        except Exception as e:
            logger.error("Failed to create pipeline with config %s: %s", config, e)
            raise e