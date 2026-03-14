from typing import Dict, List, Optional
import logging

from .base import BaseEvaluator, EvaluationResult
from .metrics import (
    AnswerSimilarity,
    CitationQuality,
    ContextRecall,
    Faithfulness,
    Groundedness,
    RetrievalCoverageProxy,
)

logger = logging.getLogger(__name__)


def _build_error_type(*, retrieval_score: float, grounding_score: float, citation_score: float | None = None) -> str:
    if retrieval_score < 0.2:
        return "RETRIEVAL_FAILURE"
    if grounding_score < 0.5:
        return "GROUNDING_FAILURE"
    if citation_score is not None and citation_score < 1.0:
        return "CITATION_ERROR"
    if retrieval_score < 0.5:
        return "PARTIAL_RETRIEVAL"
    return "NONE"


class Case1Evaluator(BaseEvaluator):
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.retrieval_metric = ContextRecall()
        self.answer_metric = AnswerSimilarity()
        self.faithfulness_metric = Faithfulness()
        self.weights = weights or {
            "context_recall": 0.45,
            "answer_similarity": 0.35,
            "faithfulness": 0.20,
        }

    def evaluate(
        self,
        query: str,
        retrieved_context: List[str],
        generated_answer: str,
        reference_context: Optional[str] = None,
        reference_answer: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        context_recall = self.retrieval_metric.compute(
            retrieved_context=retrieved_context,
            reference_context=reference_context or "",
        )
        answer_similarity = self.answer_metric.compute(
            generated_answer=generated_answer,
            reference_answer=reference_answer or "",
        )
        faithfulness = self.faithfulness_metric.compute(
            generated_answer=generated_answer,
            retrieved_context=retrieved_context,
        )

        metrics = {
            "context_recall": context_recall,
            "answer_similarity": answer_similarity,
            "faithfulness": faithfulness,
        }
        score = sum(metrics[name] * self.weights[name] for name in self.weights)
        error_type = _build_error_type(
            retrieval_score=context_recall,
            grounding_score=faithfulness,
        )
        diagnostic_note = (
            f"context_recall={context_recall:.2f}, "
            f"similarity={answer_similarity:.2f}, faithfulness={faithfulness:.2f}"
        )
        return EvaluationResult(
            score=score,
            metrics=metrics,
            feedback=diagnostic_note,
            error_type=error_type,
            diagnostic_note=diagnostic_note,
        )


class Case2Evaluator(BaseEvaluator):
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_judge_style: bool = True,
        strict_judge: bool = False,
    ):
        self.coverage_metric = RetrievalCoverageProxy()
        self.groundedness_metric = Groundedness()
        self.citation_metric = CitationQuality()
        self.use_judge_style = use_judge_style
        self.strict_judge = strict_judge
        self.weights = weights or {
            "retrieval_coverage_proxy": 0.45,
            "groundedness": 0.35,
            "citation_quality": 0.20,
        }

    def evaluate(
        self,
        query: str,
        retrieved_doc_ids: List[str],
        retrieved_context: List[str],
        generated_answer: str,
        reference_doc_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        retrieval_coverage = self.coverage_metric.compute(
            retrieved_doc_ids=retrieved_doc_ids,
            reference_doc_ids=reference_doc_ids or [],
        )
        groundedness = self.groundedness_metric.compute(
            query=query if self.use_judge_style else None,
            generated_answer=generated_answer,
            retrieved_context=retrieved_context,
            strict_mode=self.strict_judge,
        )
        citation_quality = self.citation_metric.compute(
            query=query if self.use_judge_style else None,
            generated_answer=generated_answer,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_context=retrieved_context if self.use_judge_style else None,
            strict_mode=self.strict_judge,
        )
        metrics = {
            "retrieval_coverage_proxy": retrieval_coverage,
            "groundedness": groundedness,
            "citation_quality": citation_quality,
        }
        score = sum(metrics[name] * self.weights[name] for name in self.weights)
        error_type = _build_error_type(
            retrieval_score=retrieval_coverage,
            grounding_score=groundedness,
            citation_score=citation_quality,
        )
        diagnostic_note = (
            f"coverage={retrieval_coverage:.2f}, "
            f"groundedness={groundedness:.2f}, citation_quality={citation_quality:.2f}"
        )
        return EvaluationResult(
            score=score,
            metrics=metrics,
            feedback=diagnostic_note,
            error_type=error_type,
            diagnostic_note=diagnostic_note,
        )


def get_evaluator(case_type: str, settings: Optional[Dict[str, Dict[str, float]]] = None):
    settings = settings or {}
    if case_type == "case1":
        return Case1Evaluator(weights=settings.get("weights"))
    if case_type == "case2":
        return Case2Evaluator(
            weights=settings.get("weights"),
            use_judge_style=settings.get("judge_style", True),
            strict_judge=settings.get("strict_judge", False),
        )
    raise ValueError(f"Unknown case type: {case_type}")