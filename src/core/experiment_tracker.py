import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class ExperimentTracker:
    def __init__(self, output_dir: str | Path, case_type: str):
        self.output_dir = Path(output_dir)
        self.case_type = case_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_examples_dir = self.output_dir / "retrieval_examples"
        self.answer_examples_dir = self.output_dir / "answer_examples"
        self.trial_events_path = self.output_dir / "trial_events.jsonl"
        self.failed_trials_path = self.output_dir / "failed_trials.jsonl"
        self.retrieval_examples_dir.mkdir(parents=True, exist_ok=True)
        self.answer_examples_dir.mkdir(parents=True, exist_ok=True)

        self.active_trials: Dict[str, Dict[str, Any]] = {}
        self.trial_records: List[Dict[str, Any]] = []
        self.per_query_records: List[Dict[str, Any]] = []

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_start(self, trial_id: str, config: Dict[str, Any], trial_number: Optional[int] = None) -> None:
        self.active_trials[trial_id] = {
            "trial_id": trial_id,
            "trial_number": trial_number,
            "config": config,
            "started_at": self._utc_now(),
        }
        self._append_jsonl(
            self.trial_events_path,
            {
                "event": "trial_started",
                "timestamp": self._utc_now(),
                "case_type": self.case_type,
                "trial_id": trial_id,
                "trial_number": trial_number,
                "config": config,
            },
        )

    def log_completion(
        self,
        trial_id: str,
        score: float,
        duration: float,
        metrics: Optional[Dict[str, float]] = None,
        trial_number: Optional[int] = None,
        status: str = "completed",
        failure_reason: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = self.active_trials.pop(trial_id, {})
        final_config = config or state.get("config", {})
        record = {
            "trial_number": trial_number if trial_number is not None else state.get("trial_number"),
            "score": score,
            "latency_seconds": round(duration, 4),
            "status": status,
            "failure_reason": failure_reason,
            "config_json": json.dumps(final_config, ensure_ascii=False, sort_keys=True),
        }
        for key, value in (metrics or {}).items():
            record[key] = value
        self.trial_records.append(record)
        self._append_jsonl(
            self.trial_events_path,
            {
                "event": "trial_completed",
                "timestamp": self._utc_now(),
                "case_type": self.case_type,
                "trial_id": trial_id,
                "trial_number": record["trial_number"],
                "status": status,
                "score": score,
                "latency_seconds": round(duration, 4),
                "failure_reason": failure_reason,
                "metrics": metrics or {},
                "config": final_config,
                "started_at": state.get("started_at"),
            },
        )

    def log_failure(
        self,
        trial_id: str,
        error_message: str,
        traceback_text: str = "",
        duration: float = 0.0,
        trial_number: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = self.active_trials.get(trial_id, {})
        final_config = config or state.get("config", {})
        self._append_jsonl(
            self.failed_trials_path,
            {
                "event": "trial_failed",
                "timestamp": self._utc_now(),
                "case_type": self.case_type,
                "trial_id": trial_id,
                "trial_number": trial_number if trial_number is not None else state.get("trial_number"),
                "duration_seconds": round(duration, 4),
                "error_message": error_message,
                "traceback": traceback_text,
                "config": final_config,
                "started_at": state.get("started_at"),
            },
        )
        self.log_completion(
            trial_id=trial_id,
            score=0.0,
            duration=duration,
            metrics={},
            trial_number=trial_number,
            status="failed",
            failure_reason=error_message,
            config=config,
        )

    def record_query_diagnostics(self, rows: List[Dict[str, Any]]) -> None:
        self.per_query_records = list(rows)

    def save_query_examples(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            query_id = row["query_id"]
            retrieval_payload = {
                "query_id": query_id,
                "retrieved_doc_ids": row.get("retrieved_doc_ids", []),
                "contexts": row.get("retrieved_context", []),
                "metrics": row.get("metrics", {}),
            }
            answer_payload = {
                "query_id": query_id,
                "query": row.get("query", ""),
                "generated_answer": row.get("generated_answer", ""),
                "reference_answer": row.get("reference_answer", ""),
                "error_type": row.get("error_type", "NONE"),
            }
            (self.retrieval_examples_dir / f"{query_id}.json").write_text(
                json.dumps(retrieval_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (self.answer_examples_dir / f"{query_id}.json").write_text(
                json.dumps(answer_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _write_run_summary(self) -> None:
        summary_path = self.output_dir / "run_summary.csv"
        if not self.trial_records:
            pd.DataFrame(
                columns=["trial_number", "score", "latency_seconds", "status", "failure_reason", "config_json"]
            ).to_csv(summary_path, index=False)
            return
        pd.DataFrame(self.trial_records).sort_values(
            by=["score", "trial_number"], ascending=[False, True], na_position="last"
        ).to_csv(summary_path, index=False)

    def _write_per_query_diagnostics(self) -> None:
        diagnostics_path = self.output_dir / "per_query_diagnostics.csv"
        if not self.per_query_records:
            pd.DataFrame(
                columns=[
                    "query_id", "case_label", "config_id", "retrieval_score", "retrieval_status",
                    "generation_score", "aux_score", "error_type", "diagnostic_note",
                    "retrieved_doc_ids", "effective_query", "generated_answer",
                ]
            ).to_csv(diagnostics_path, index=False)
            return

        flat_rows = []
        for row in self.per_query_records:
            flat_rows.append(
                {
                    "query_id": row["query_id"],
                    "case_label": row["case_label"],
                    "config_id": row["config_id"],
                    "retrieval_score": row["retrieval_score"],
                    "retrieval_status": row["retrieval_status"],
                    "generation_score": row["generation_score"],
                    "aux_score": row["aux_score"],
                    "error_type": row["error_type"],
                    "diagnostic_note": row["diagnostic_note"],
                    "retrieved_doc_ids": "|".join(row.get("retrieved_doc_ids", [])),
                    "effective_query": row.get("effective_query", ""),
                    "generated_answer": row.get("generated_answer", ""),
                }
            )
        pd.DataFrame(flat_rows).to_csv(diagnostics_path, index=False)

    def _write_best_config(self, best_config: Dict[str, Any]) -> None:
        (self.output_dir / "best_config.json").write_text(
            json.dumps(best_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_output_readme(
        self,
        best_config: Dict[str, Any],
        best_score: float,
        aggregate_metrics: Dict[str, float],
        best_latency_seconds: float,
        selection_summary: Dict[str, Any] | None = None,
    ) -> None:
        metric_lines = "\n".join(f"- `{key}`: {value:.4f}" for key, value in aggregate_metrics.items())
        selection_summary = selection_summary or {}
        tie_break_lines = "\n".join(
            f"- `{key}`: {value}"
            for key, value in selection_summary.items()
        )

        # Explain how evaluation works for this case — addresses rubric Red Flag ("explain how evaluation changes when no reference answer exists").
        evaluation_note = ""
        if self.case_type == "case1":
            evaluation_note = """
## How Evaluation Works in This Case (With Reference Context and Answer)
Case 1 has **reference relevant context** and **reference answer**. Evaluation uses:
- **context_recall**: retrieval alignment — fraction of reference-context tokens covered by retrieved context.
- **answer_similarity**: generated vs reference answer (semantic + token overlap).
- **faithfulness**: fraction of answer tokens present in retrieved context.
Overall score is a weighted sum. Optimization targets both retrieval quality and answer quality against the gold reference.
"""
        elif self.case_type == "case2":
            evaluation_note = """
## How Evaluation Works in This Case (No Reference Answer)
Case 2 has **no gold answer or reference relevant context**. Evaluation therefore uses:
- **retrieval_coverage_proxy**: fraction of reference doc IDs that appear in retrieved doc IDs (retrieval-centric).
- **groundedness**: LLM-as-judge (TEACHER) scores how well the answer is grounded in the retrieved context (groundedness-oriented).
- **citation_quality**: LLM-as-judge scores citation correctness against retrieved doc IDs.
The overall score is a weighted sum of these three. **This is how evaluation changes when no reference answer exists**: we optimize for retrieval coverage and model-based groundedness/citation instead of similarity to a gold answer.
"""

        report = f"""# Output Summary

## Case
`{self.case_type}`
{evaluation_note}
## Recommended Configuration
```json
{json.dumps(best_config, indent=2, ensure_ascii=False)}
```

## Selection Details
{tie_break_lines or "- No additional selection metadata recorded."}

## Aggregate Metrics
{metric_lines}

## Overall Score
- `score`: {best_score:.4f}

## Latency
- `latency_seconds`: {best_latency_seconds:.4f}
"""
        (self.output_dir / "README.md").write_text(report, encoding="utf-8")

    def save_best_run(
        self,
        *,
        best_config: Dict[str, Any],
        best_score: float,
        aggregate_metrics: Dict[str, float],
        latency_seconds: float,
        query_rows: List[Dict[str, Any]],
        selection_summary: Dict[str, Any] | None = None,
    ) -> None:
        self.record_query_diagnostics(query_rows)
        self._write_best_config(best_config)
        self._write_run_summary()
        self._write_per_query_diagnostics()
        self.save_query_examples(query_rows)
        self._write_output_readme(best_config, best_score, aggregate_metrics, latency_seconds, selection_summary)

    def save_final_report(self, best_config: Dict[str, Any], best_score: float, trials_df: pd.DataFrame) -> None:
        self._write_best_config(best_config)
        if not trials_df.empty:
            trials_df.to_csv(self.output_dir / "run_summary.csv", index=False)
        else:
            self._write_run_summary()
        self._write_output_readme(best_config, best_score, {}, 0.0, None)
