from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Chunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    doc_id: str


@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    scores: List[float]
    query: str
    effective_query: str
    retriever_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def doc_ids(self) -> List[str]:
        seen: List[str] = []
        for chunk in self.chunks:
            if chunk.doc_id not in seen:
                seen.append(chunk.doc_id)
        return seen

    @property
    def contexts(self) -> List[str]:
        return [chunk.content for chunk in self.chunks]


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Split documents into chunks."""


class BaseRetriever(ABC):
    @abstractmethod
    def index(self, chunks: List[Chunk]) -> None:
        """Index chunks into an internal retrieval structure."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve relevant chunks for the given query."""


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, result: RetrievalResult, top_n: int = 5) -> RetrievalResult:
        """Rerank retrieved chunks."""


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, retrieval_result: RetrievalResult) -> str:
        """Generate an answer from retrieved chunks."""


class BaseQueryRefiner(ABC):
    @abstractmethod
    def refine(self, query: str) -> str:
        """Rewrite a query into a retrieval-friendly form."""