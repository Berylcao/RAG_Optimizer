import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.run_optimization import build_selection_summary, select_best_trial
from src.core.experiment_tracker import ExperimentTracker
from src.pipelines.factory import PipelineFactory


class SelectionAndTrackingTests(unittest.TestCase):
    def test_student_direct_uses_env_model_independent_of_provider(self) -> None:
        config = {
            "chunking": {"size": 128, "overlap": 0, "strategy": "token"},
            "indexing": {"retriever": "bm25", "embedding_model": "bge-small", "metadata_enrichment": False},
            "reranking": {"enabled": False},
            "query_refinement": {"rewrite": False, "decompose": False},
            "generation": {"llm": "student-direct", "temperature": 0.0, "answer_style": "concise"},
        }
        corpus = [{"doc_id": "D1", "text": "Premium users may be exempt."}]

        with patch(
            "src.pipelines.factory.get_model_config",
            return_value={
                "raw_model_name": "gpt-4o-mini",
                "api_key": "sk-demo",
                "api_base": "",
            },
        ):
            pipeline = PipelineFactory.create_pipeline(config, corpus, strict_formal_mode=True)

        self.assertEqual(type(pipeline.generator).__name__, "StudentChatGenerator")
        self.assertEqual(pipeline.generator.llm_name, "student-direct")

    def test_tie_break_prefers_quality_then_latency(self) -> None:
        trials = [
            SimpleNamespace(
                number=2,
                value=0.8,
                user_attrs={
                    "aggregate_metrics": {
                        "context_recall": 0.90,
                        "answer_similarity": 0.60,
                        "faithfulness": 0.95,
                    },
                    "latency_seconds": 15.0,
                    "config": {"name": "slower"},
                },
            ),
            SimpleNamespace(
                number=1,
                value=0.8,
                user_attrs={
                    "aggregate_metrics": {
                        "context_recall": 0.92,
                        "answer_similarity": 0.60,
                        "faithfulness": 0.95,
                    },
                    "latency_seconds": 20.0,
                    "config": {"name": "better_retrieval"},
                },
            ),
            SimpleNamespace(
                number=0,
                value=0.8,
                user_attrs={
                    "aggregate_metrics": {
                        "context_recall": 0.92,
                        "answer_similarity": 0.60,
                        "faithfulness": 0.95,
                    },
                    "latency_seconds": 10.0,
                    "config": {"name": "faster"},
                },
            ),
        ]

        best_trial = select_best_trial(trials, "case1")
        selection_summary = build_selection_summary(best_trial, "case1")

        self.assertEqual(best_trial.number, 0)
        self.assertEqual(selection_summary["context_recall"], 0.92)
        self.assertEqual(selection_summary["latency_seconds"], 10.0)

    def test_tracker_writes_event_and_failure_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(output_dir=Path(temp_dir), case_type="case2")

            tracker.log_start("trial-1", {"generation": {"llm": "student-direct"}}, trial_number=0)
            tracker.log_failure(
                "trial-1",
                error_message="teacher judge timeout",
                traceback_text="Traceback: timeout",
                duration=12.34,
                trial_number=0,
                config={"generation": {"llm": "student-direct"}},
            )

            events = [json.loads(line) for line in (Path(temp_dir) / "trial_events.jsonl").read_text(encoding="utf-8").splitlines()]
            failures = [json.loads(line) for line in (Path(temp_dir) / "failed_trials.jsonl").read_text(encoding="utf-8").splitlines()]

            self.assertEqual(events[0]["event"], "trial_started")
            self.assertEqual(events[1]["event"], "trial_completed")
            self.assertEqual(events[1]["status"], "failed")
            self.assertEqual(failures[0]["event"], "trial_failed")
            self.assertIn("timeout", failures[0]["traceback"])


if __name__ == "__main__":
    unittest.main()
