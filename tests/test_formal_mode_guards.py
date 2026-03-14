import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.evaluation.evaluators import Case2Evaluator
from src.pipelines.base import Chunk, RetrievalResult
from src.pipelines.components import StudentChatGenerator
from src.pipelines.factory import PipelineFactory


class FormalModeGuardTests(unittest.TestCase):
    def test_decompose_only_does_not_trigger_rewrite(self) -> None:
        config = {
            "chunking": {"size": 128, "overlap": 0, "strategy": "token"},
            "indexing": {"retriever": "bm25", "embedding_model": "bge-small", "metadata_enrichment": False},
            "reranking": {"enabled": False},
            "query_refinement": {"rewrite": False, "decompose": True},
            "generation": {"llm": "student-direct", "temperature": 0.0, "answer_style": "concise"},
        }
        corpus = [{"doc_id": "D1", "text": "KYC and AML requirements are documented here."}]

        pipeline = PipelineFactory.create_pipeline(config, corpus, strict_formal_mode=True)
        result = pipeline.retrieve("KYC and AML requirements", top_k=1)

        self.assertEqual(
            result.effective_query,
            "KYC and AML requirements KYC AML requirements",
        )
        self.assertNotIn("Know Your Customer", result.effective_query)

    def test_student_generator_raises_in_strict_formal_mode(self) -> None:
        generator = StudentChatGenerator(
            llm_name="qwen3.5-35b-a3b",
            answer_style="citation_first",
            strict_mode=True,
        )
        retrieval_result = RetrievalResult(
            chunks=[Chunk(chunk_id="C1", content="Premium users may be exempt.", metadata={}, doc_id="D1")],
            scores=[1.0],
            query="Which users are exempt?",
            effective_query="Which users are exempt?",
            retriever_name="bm25",
        )

        with patch("src.pipelines.components.get_model_config", return_value={"api_key": "k", "api_base": "https://example.com", "raw_model_name": "demo-model"}):
            with patch("src.pipelines.components.OpenAI", side_effect=RuntimeError("network down")):
                with self.assertRaises(RuntimeError):
                    generator.generate("Which users are exempt?", retrieval_result)

    def test_case2_strict_judge_raises_on_judge_failure(self) -> None:
        evaluator = Case2Evaluator(use_judge_style=True, strict_judge=True)

        with patch("src.evaluation.metrics.dspy_ready", return_value=True):
            with patch("src.evaluation.metrics.judge_groundedness_with_dspy", side_effect=RuntimeError("judge unavailable")):
                with self.assertRaises(RuntimeError):
                    evaluator.evaluate(
                        query="What are the fee waiver conditions?",
                        retrieved_doc_ids=["D1"],
                        retrieved_context=["Premium users may receive a waiver."],
                        generated_answer="Premium users may receive a waiver. [D1]",
                        reference_doc_ids=["D1"],
                    )


if __name__ == "__main__":
    unittest.main()
