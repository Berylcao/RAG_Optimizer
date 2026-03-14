import re
from typing import List

from src.utils.dspy_model_loader import configure_dspy_lm, dspy, dspy_is_installed


def dspy_ready(model_type: str = "TEACHER") -> bool:
    if not dspy_is_installed():
        return False
    return configure_dspy_lm(model_type)


if dspy is not None:
    def _parse_score(raw_value: str | float | int, default: float = 0.0) -> float:
        if isinstance(raw_value, (int, float)):
            return max(0.0, min(1.0, float(raw_value)))
        match = re.search(r"([01](?:\.\d+)?)", str(raw_value))
        if not match:
            return default
        return max(0.0, min(1.0, float(match.group(1))))


    class QueryRewriteSignature(dspy.Signature):
        """Transform a user query into a more optimized search query for retrieval."""

        original_query = dspy.InputField(desc="The user's original question.")
        rewritten_query = dspy.OutputField(desc="An optimized query for search.")


    class RAGGenerationSignature(dspy.Signature):
        """Answer the question based on the provided context."""

        context = dspy.InputField(desc="Retrieved context documents.")
        question = dspy.InputField()
        answer = dspy.OutputField(desc="A concise and grounded answer.")


    class GroundednessJudgeSignature(dspy.Signature):
        """Score how well an answer is grounded in the provided context."""

        question = dspy.InputField(desc="The original user question.")
        context = dspy.InputField(desc="Retrieved context with evidence.")
        answer = dspy.InputField(desc="Candidate answer to judge.")
        score = dspy.OutputField(desc="A single numeric score between 0.0 and 1.0.")


    class CitationJudgeSignature(dspy.Signature):
        """Score citation quality using the question, answer, and retrieved evidence."""

        question = dspy.InputField(desc="The original user question.")
        available_doc_ids = dspy.InputField(desc="Pipe-delimited list of retrieved document ids.")
        context = dspy.InputField(desc="Retrieved context with explicit document identifiers.")
        answer = dspy.InputField(desc="Candidate answer with citations to judge.")
        score = dspy.OutputField(desc="A single numeric score between 0.0 and 1.0.")


    class DSPyQueryRewriteModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.rewriter = dspy.ChainOfThought(QueryRewriteSignature)

        def forward(self, original_query):
            return self.rewriter(original_query=original_query)


    class DSPyRAGModule(dspy.Module):
        def __init__(self, use_rewrite: bool = True):
            super().__init__()
            self.use_rewrite = use_rewrite
            self.rewriter = DSPyQueryRewriteModule() if use_rewrite else None
            self.responder = dspy.ChainOfThought(RAGGenerationSignature)

        def forward(self, question, context):
            final_query = question
            if self.use_rewrite and self.rewriter is not None:
                rewrite_result = self.rewriter(original_query=question)
                final_query = rewrite_result.rewritten_query
            context_text = "\n".join(context)
            prediction = self.responder(context=context_text, question=final_query)
            return dspy.Prediction(answer=prediction.answer, rewritten_query=final_query)


    class GroundednessJudgeModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.judge = dspy.ChainOfThought(GroundednessJudgeSignature)

        def forward(self, question, context, answer):
            return self.judge(question=question, context=context, answer=answer)


    class CitationJudgeModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.judge = dspy.ChainOfThought(CitationJudgeSignature)

        def forward(self, question, available_doc_ids, context, answer):
            return self.judge(
                question=question,
                available_doc_ids=available_doc_ids,
                context=context,
                answer=answer,
            )


    def rewrite_query_with_dspy(query: str, model_type: str = "TEACHER") -> str:
        if not dspy_ready(model_type):
            raise RuntimeError("DSPy is not configured.")
        module = DSPyQueryRewriteModule()
        prediction = module(original_query=query)
        return prediction.rewritten_query


    def generate_answer_with_dspy(query: str, contexts: List[str], model_type: str = "STUDENT") -> str:
        if not dspy_ready(model_type):
            raise RuntimeError("DSPy is not configured.")
        module = DSPyRAGModule(use_rewrite=False)
        prediction = module(question=query, context=contexts)
        return prediction.answer


    def judge_groundedness_with_dspy(
        query: str,
        contexts: List[str],
        answer: str,
        model_type: str = "TEACHER",
    ) -> float:
        if not dspy_ready(model_type):
            raise RuntimeError("DSPy is not configured.")
        module = GroundednessJudgeModule()
        prediction = module(question=query, context="\n".join(contexts), answer=answer)
        return _parse_score(prediction.score)


    def judge_citation_quality_with_dspy(
        query: str,
        contexts: List[str],
        answer: str,
        retrieved_doc_ids: List[str],
        model_type: str = "TEACHER",
    ) -> float:
        if not dspy_ready(model_type):
            raise RuntimeError("DSPy is not configured.")
        module = CitationJudgeModule()
        context_text = "\n".join(contexts)
        available_doc_ids = "|".join(retrieved_doc_ids)
        prediction = module(
            question=query,
            available_doc_ids=available_doc_ids,
            context=context_text,
            answer=answer,
        )
        return _parse_score(prediction.score)
else:
    class DSPyRAGModule:  # pragma: no cover - optional dependency stub
        def __init__(self, use_rewrite: bool = True):
            self.use_rewrite = use_rewrite


    def rewrite_query_with_dspy(query: str, model_type: str = "TEACHER") -> str:  # pragma: no cover
        raise RuntimeError("DSPy is not installed.")


    def generate_answer_with_dspy(query: str, contexts: List[str], model_type: str = "STUDENT") -> str:  # pragma: no cover
        raise RuntimeError("DSPy is not installed.")


    def judge_groundedness_with_dspy(
        query: str,
        contexts: List[str],
        answer: str,
        model_type: str = "TEACHER",
    ) -> float:  # pragma: no cover
        raise RuntimeError("DSPy is not installed.")


    def judge_citation_quality_with_dspy(
        query: str,
        contexts: List[str],
        answer: str,
        retrieved_doc_ids: List[str],
        model_type: str = "TEACHER",
    ) -> float:  # pragma: no cover
        raise RuntimeError("DSPy is not installed.")
