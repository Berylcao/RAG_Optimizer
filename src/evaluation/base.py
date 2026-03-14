from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    score: float
    metrics: Dict[str, float]
    feedback: str
    error_type: str = "NONE"
    diagnostic_note: str = ""
    details: Dict[str, Any] | None = None


class BaseMetric(ABC):
    """Abstract Base Class for all RAG metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute(self, **kwargs) -> float:
        """Compute the metric score."""
        raise NotImplementedError


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, query: str, retrieved_context: List[str], generated_answer: str, **kwargs) -> EvaluationResult:
        """Aggregate per-query metrics into a single optimization score."""