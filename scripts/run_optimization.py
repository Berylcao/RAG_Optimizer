import argparse
import json
import logging
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List

import optuna
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.core.experiment_tracker import ExperimentTracker
from src.evaluation.evaluators import get_evaluator
from src.pipelines.factory import PipelineFactory
from src.utils.data_loader import load_data

logger = logging.getLogger(__name__)


def resolve_path(path_str: str, *, default_base: Path = ROOT_DIR) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (default_base / path).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_nested_value(config: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def sample_parameter(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    param_type = spec.get("type", "categorical")
    if param_type == "fixed":
        return spec["value"]
    if param_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if param_type == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if param_type == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
    raise ValueError(f"Unsupported parameter type: {param_type}")


def sample_config(trial: optuna.Trial, search_space: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for section_name, section_spec in search_space.items():
        config[section_name] = {}
        for param_name, param_spec in section_spec.items():
            depends_on = param_spec.get("depends_on", {})
            if depends_on:
                unmet_dependency = False
                for dep_key, dep_value in depends_on.items():
                    if get_nested_value(config, dep_key) != dep_value:
                        unmet_dependency = True
                        break
                if unmet_dependency:
                    continue
            optuna_name = f"{section_name}.{param_name}"
            config[section_name][param_name] = sample_parameter(trial, optuna_name, param_spec)
    return config


def aggregate_metric_rows(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_rows:
        return {}
    keys = metric_rows[0].keys()
    return {key: sum(row.get(key, 0.0) for row in metric_rows) / len(metric_rows) for key in keys}


def retrieval_status_from_score(score: float) -> str:
    if score >= 0.8:
        return "SUCCESS"
    if score >= 0.5:
        return "PARTIAL"
    return "MISS"


def metric_aliases(case_type: str) -> Dict[str, str]:
    if case_type == "case1":
        return {
            "retrieval_score": "context_recall",
            "generation_score": "answer_similarity",
            "aux_score": "faithfulness",
        }
    return {
        "retrieval_score": "retrieval_coverage_proxy",
        "generation_score": "groundedness",
        "aux_score": "citation_quality",
    }


def ranking_metrics(case_type: str) -> List[str]:
    if case_type == "case1":
        return ["context_recall", "answer_similarity", "faithfulness"]
    return ["groundedness", "citation_quality", "retrieval_coverage_proxy"]


def build_trial_ranking_key(trial: optuna.trial.FrozenTrial, case_type: str) -> tuple:
    metrics = trial.user_attrs.get("aggregate_metrics", {})
    metric_values = tuple(round(float(metrics.get(name, 0.0)), 6) for name in ranking_metrics(case_type))
    score = round(float(trial.value if trial.value is not None else 0.0), 6)
    latency = round(float(trial.user_attrs.get("latency_seconds", float("inf"))), 6)
    trial_number = int(trial.number)
    return (score, *metric_values, -latency, -trial_number)


def select_best_trial(completed_trials: List[optuna.trial.FrozenTrial], case_type: str) -> optuna.trial.FrozenTrial:
    return sorted(completed_trials, key=lambda trial: build_trial_ranking_key(trial, case_type), reverse=True)[0]


def build_selection_summary(trial: optuna.trial.FrozenTrial, case_type: str) -> Dict[str, Any]:
    metrics = trial.user_attrs.get("aggregate_metrics", {})
    summary: Dict[str, Any] = {
        "selection_rule": "overall score -> case-specific quality metrics -> lower latency -> lower trial number",
        "selected_trial_number": trial.number,
        "selected_score": round(float(trial.value if trial.value is not None else 0.0), 4),
        "latency_seconds": round(float(trial.user_attrs.get("latency_seconds", 0.0)), 4),
    }
    for metric_name in ranking_metrics(case_type):
        summary[metric_name] = round(float(metrics.get(metric_name, 0.0)), 4)
    return summary


def evaluate_trial(
    *,
    trial: optuna.Trial,
    case_type: str,
    config: Dict[str, Any],
    dataset,
    evaluator,
    tracker: ExperimentTracker,
    top_k: int,
    failure_score: float,
    strict_formal_mode: bool,
) -> float:
    trial_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    tracker.log_start(trial_id, config, trial.number)
    aliases = metric_aliases(case_type)

    try:
        pipeline = PipelineFactory.create_pipeline(
            config,
            dataset.get_corpus_for_indexing(),
            strict_formal_mode=strict_formal_mode,
        )
        query_rows: List[Dict[str, Any]] = []
        query_scores: List[float] = []
        metric_rows: List[Dict[str, float]] = []

        for step, item in enumerate(dataset.get_eval_dataset()):
            retrieval_result = pipeline.retrieve(item["query"], top_k=top_k)
            generated_answer = pipeline.generate(item["query"], retrieval_result)

            if case_type == "case1":
                evaluation = evaluator.evaluate(
                    query=item["query"],
                    retrieved_context=retrieval_result.contexts,
                    generated_answer=generated_answer,
                    reference_context=item.get("reference_context", ""),
                    reference_answer=item.get("reference_answer", ""),
                )
            else:
                evaluation = evaluator.evaluate(
                    query=item["query"],
                    retrieved_doc_ids=retrieval_result.doc_ids,
                    retrieved_context=retrieval_result.contexts,
                    generated_answer=generated_answer,
                    reference_doc_ids=item.get("reference_doc_ids", []),
                )

            query_scores.append(evaluation.score)
            metric_rows.append(evaluation.metrics)
            query_rows.append(
                {
                    "query_id": item["query_id"],
                    "case_label": case_type,
                    "config_id": f"trial_{trial.number}",
                    "query": item["query"],
                    "reference_answer": item.get("reference_answer", ""),
                    "retrieval_score": evaluation.metrics[aliases["retrieval_score"]],
                    "retrieval_status": retrieval_status_from_score(evaluation.metrics[aliases["retrieval_score"]]),
                    "generation_score": evaluation.metrics[aliases["generation_score"]],
                    "aux_score": evaluation.metrics[aliases["aux_score"]],
                    "error_type": evaluation.error_type,
                    "diagnostic_note": evaluation.diagnostic_note,
                    "retrieved_doc_ids": retrieval_result.doc_ids,
                    "effective_query": retrieval_result.effective_query,
                    "generated_answer": generated_answer,
                    "retrieved_context": retrieval_result.contexts,
                    "metrics": evaluation.metrics,
                }
            )

            running_score = sum(query_scores) / len(query_scores)
            trial.report(running_score, step=step)
            if trial.should_prune():
                duration = time.time() - start_time
                tracker.log_completion(
                    trial_id=trial_id,
                    score=running_score,
                    duration=duration,
                    metrics=aggregate_metric_rows(metric_rows),
                    trial_number=trial.number,
                    status="pruned",
                    failure_reason="Trial pruned by Optuna",
                    config=config,
                )
                raise optuna.TrialPruned()

        aggregate_metrics = aggregate_metric_rows(metric_rows)
        final_score = sum(query_scores) / len(query_scores) if query_scores else failure_score
        duration = time.time() - start_time

        tracker.log_completion(
            trial_id=trial_id,
            score=final_score,
            duration=duration,
            metrics=aggregate_metrics,
            trial_number=trial.number,
            config=config,
        )

        trial.set_user_attr("config", config)
        trial.set_user_attr("aggregate_metrics", aggregate_metrics)
        trial.set_user_attr("latency_seconds", round(duration, 4))
        trial.set_user_attr("query_rows", query_rows)
        return final_score

    except optuna.TrialPruned:
        raise
    except Exception as exc:
        duration = time.time() - start_time
        tracker.log_failure(
            trial_id=trial_id,
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
            duration=duration,
            trial_number=trial.number,
            config=config,
        )
        logger.exception("Trial %s failed.", trial.number)
        return failure_score


def _evaluate_on_holdout(
    *,
    best_config: Dict[str, Any],
    holdout_dataset,
    evaluator,
    case_type: str,
    top_k: int,
    strict_formal_mode: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Re-evaluate the winning config on the holdout split."""
    aliases = metric_aliases(case_type)
    pipeline = PipelineFactory.create_pipeline(
        best_config,
        holdout_dataset.get_corpus_for_indexing(),
        strict_formal_mode=strict_formal_mode,
    )
    rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, float]] = []
    for item in holdout_dataset.get_eval_dataset():
        retrieval_result = pipeline.retrieve(item["query"], top_k=top_k)
        generated_answer = pipeline.generate(item["query"], retrieval_result)
        if case_type == "case1":
            evaluation = evaluator.evaluate(
                query=item["query"],
                retrieved_context=retrieval_result.contexts,
                generated_answer=generated_answer,
                reference_context=item.get("reference_context", ""),
                reference_answer=item.get("reference_answer", ""),
            )
        else:
            evaluation = evaluator.evaluate(
                query=item["query"],
                retrieved_doc_ids=retrieval_result.doc_ids,
                retrieved_context=retrieval_result.contexts,
                generated_answer=generated_answer,
                reference_doc_ids=item.get("reference_doc_ids", []),
            )
        metric_rows.append(evaluation.metrics)
        rows.append({
            "query_id": item["query_id"],
            "case_label": case_type,
            "config_id": "best_holdout",
            "query": item["query"],
            "reference_answer": item.get("reference_answer", ""),
            "retrieval_score": evaluation.metrics[aliases["retrieval_score"]],
            "retrieval_status": retrieval_status_from_score(evaluation.metrics[aliases["retrieval_score"]]),
            "generation_score": evaluation.metrics[aliases["generation_score"]],
            "aux_score": evaluation.metrics[aliases["aux_score"]],
            "error_type": evaluation.error_type,
            "diagnostic_note": f"[holdout] {evaluation.diagnostic_note}",
            "retrieved_doc_ids": retrieval_result.doc_ids,
            "effective_query": retrieval_result.effective_query,
            "generated_answer": generated_answer,
            "retrieved_context": retrieval_result.contexts,
            "metrics": evaluation.metrics,
            "score": evaluation.score,
        })
    holdout_metrics = aggregate_metric_rows(metric_rows)
    return rows, holdout_metrics


def _compute_param_importance(study: optuna.Study) -> Dict[str, float]:
    """Return Optuna hyperparameter importance; empty dict on failure."""
    try:
        from optuna.importance import get_param_importances
        importance = get_param_importances(study)
        return {k: round(v, 4) for k, v in importance.items()}
    except Exception:
        logger.debug("Parameter importance calculation skipped (not enough trials or unsupported).")
        return {}


def run_optimization(
    *,
    case_type: str,
    data_dir: Path,
    search_space_path: Path,
    evaluation_settings_path: Path,
    output_root: Path,
    n_trials: int,
    timeout: int | None,
    max_queries: int | None = None,
) -> Dict[str, Any]:
    search_space = load_yaml(search_space_path)
    evaluation_settings = load_yaml(evaluation_settings_path)
    case_settings = dict(evaluation_settings.get(case_type, {}))
    global_settings = evaluation_settings.get("global", {})
    top_k = global_settings.get("top_k", 5)
    random_seed = global_settings.get("random_seed", 42)
    holdout_fraction = global_settings.get("holdout_fraction", 0.4)
    failure_score = global_settings.get("optimization", {}).get("failure_score", 0.0)
    direction = global_settings.get("optimization", {}).get("direction", "maximize")
    full_dataset = load_data(
        case_type,
        str(data_dir),
        max_queries=max_queries,
        seed=random_seed,
    )
    strict_formal_mode = global_settings.get("formal_mode", {}).get("strict_remote_fallbacks", True)

    tuning_dataset, holdout_dataset = full_dataset.split(
        holdout_fraction=holdout_fraction, seed=random_seed,
    )

    if case_type == "case2":
        case_settings["strict_judge"] = strict_formal_mode and case_settings.get("judge_style", True)
    evaluator = get_evaluator(case_type, case_settings)

    output_dir = output_root / case_type
    tracker = ExperimentTracker(output_dir=output_dir, case_type=case_type)

    sampler = optuna.samplers.TPESampler(seed=random_seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=f"rag_optimizer_{case_type}",
    )

    study.optimize(
        lambda trial: evaluate_trial(
            trial=trial,
            case_type=case_type,
            config=sample_config(trial, search_space),
            dataset=tuning_dataset,
            evaluator=evaluator,
            tracker=tracker,
            top_k=top_k,
            failure_score=failure_score,
            strict_formal_mode=strict_formal_mode,
        ),
        n_trials=n_trials,
        timeout=timeout,
    )

    completed_trials = [
        trial for trial in study.trials
        if trial.value is not None and "config" in trial.user_attrs
    ]
    if not completed_trials:
        raise RuntimeError("No successful trials completed. Check run_summary.csv for failure details.")

    if direction != "maximize":
        raise ValueError("This implementation currently supports maximize-only study selection.")
    best_trial = select_best_trial(completed_trials, case_type)
    best_config = best_trial.user_attrs["config"]
    best_latency = best_trial.user_attrs.get("latency_seconds", 0.0)
    selection_summary = build_selection_summary(best_trial, case_type)

    holdout_rows, holdout_metrics = _evaluate_on_holdout(
        best_config=best_config,
        holdout_dataset=holdout_dataset,
        evaluator=evaluator,
        case_type=case_type,
        top_k=top_k,
        strict_formal_mode=strict_formal_mode,
    )
    tuning_metrics = best_trial.user_attrs.get("aggregate_metrics", {})
    selection_summary["tuning_score"] = round(float(best_trial.value or 0.0), 4)
    selection_summary["holdout_score"] = round(
        sum(r["score"] for r in holdout_rows) / max(1, len(holdout_rows)), 4
    )
    for key, value in holdout_metrics.items():
        selection_summary[f"holdout_{key}"] = round(value, 4)

    best_query_rows = best_trial.user_attrs.get("query_rows", []) + holdout_rows

    tracker.save_best_run(
        best_config=best_config,
        best_score=best_trial.value,
        aggregate_metrics=tuning_metrics,
        latency_seconds=best_latency,
        query_rows=best_query_rows,
        selection_summary=selection_summary,
    )

    param_importance = _compute_param_importance(study)
    if param_importance:
        import json as _json
        (output_dir / "param_importance.json").write_text(
            _json.dumps(param_importance, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info("Parameter importance: %s", param_importance)

    return {
        "case_type": case_type,
        "best_score": best_trial.value,
        "best_config": best_config,
        "aggregate_metrics": tuning_metrics,
        "holdout_metrics": holdout_metrics,
        "param_importance": param_importance,
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RAG optimizer end-to-end.")
    parser.add_argument("--case-type", choices=["case1", "case2"], required=True)
    parser.add_argument("--data-dir", default="data/hotpotqa", help="Data directory (default: data/hotpotqa).")
    parser.add_argument("--search-space", default="configs/search_space.yaml")
    parser.add_argument("--evaluation-settings", default="configs/evaluation_settings.yaml")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None, help="Limit dataset to N queries for faster runs.")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    args = build_parser().parse_args()
    result = run_optimization(
        case_type=args.case_type,
        data_dir=resolve_path(args.data_dir),
        search_space_path=resolve_path(args.search_space),
        evaluation_settings_path=resolve_path(args.evaluation_settings),
        output_root=resolve_path(args.output_root),
        n_trials=args.n_trials,
        timeout=args.timeout,
        max_queries=args.max_queries,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()