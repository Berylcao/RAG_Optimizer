import pandas as pd
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RAGDataset:
    """Normalized dataset wrapper for case-specific evaluation inputs."""

    def __init__(self, queries: List[Dict], corpus: List[Dict], case_type: str):
        self.queries = queries
        self.corpus = corpus
        self.case_type = case_type
        self.corpus_lookup = {doc['doc_id']: doc for doc in corpus}

    def split(
        self,
        holdout_fraction: float = 0.4,
        seed: int = 42,
    ) -> Tuple["RAGDataset", "RAGDataset"]:
        """Split into tuning and holdout sets.

        Returns (tuning_dataset, holdout_dataset).  When the dataset is too
        small (<=2 queries) the full dataset is returned as both tuning and
        holdout to keep the pipeline functional.
        """
        if len(self.queries) <= 2:
            logger.warning(
                "Dataset too small (%d queries) for a meaningful split; "
                "using full set for both tuning and holdout.",
                len(self.queries),
            )
            return self, self

        shuffled = list(self.queries)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1 - holdout_fraction)))
        tuning_queries = shuffled[:split_idx]
        holdout_queries = shuffled[split_idx:]
        if not holdout_queries:
            holdout_queries = [shuffled[-1]]
            tuning_queries = shuffled[:-1]
        logger.info(
            "Split dataset: %d tuning, %d holdout queries.",
            len(tuning_queries),
            len(holdout_queries),
        )
        return (
            RAGDataset(tuning_queries, self.corpus, self.case_type),
            RAGDataset(holdout_queries, self.corpus, self.case_type),
        )

    def get_corpus_for_indexing(self) -> List[Dict]:
        """Return raw documents for pipeline indexing."""
        return self.corpus

    def get_eval_dataset(self) -> List[Dict]:
        eval_data = []
        for q in self.queries:
            raw_doc_ids = q.get('reference_doc_ids', '')
            if isinstance(raw_doc_ids, str):
                reference_doc_ids = []
                for value in raw_doc_ids.split('|'):
                    value = value.strip()
                    if value and value not in reference_doc_ids:
                        reference_doc_ids.append(value)
            else:
                reference_doc_ids = list(dict.fromkeys(raw_doc_ids or []))

            entry = {
                "query_id": q.get('query_id'),
                "query": str(q.get('query', '')).strip(),
                "reference_doc_ids": reference_doc_ids,
            }

            if self.case_type == 'case1':
                entry['reference_answer'] = str(q.get('reference_answer', '')).strip()
                entry['reference_context'] = str(q.get('reference_relevant_context', '')).strip()

            eval_data.append(entry)
        return eval_data


def load_data(
    case_type: str,
    data_dir: str,
    max_queries: Optional[int] = None,
    seed: int = 42,
) -> RAGDataset:
    data_path = Path(data_dir)
    corpus_file = data_path / "reference_corpus.jsonl"
    corpus = []
    if corpus_file.exists():
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line.strip()))
    else:
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    if not corpus:
        raise ValueError("Corpus is empty.")

    required_corpus_cols = {"doc_id", "text"}
    for doc in corpus:
        missing = required_corpus_cols - set(doc)
        if missing:
            raise ValueError(f"Corpus document missing required keys: {sorted(missing)}")

    if case_type == 'case1':
        dataset_file = data_path / "case1_eval_dataset.csv"
        required_columns = {
            "query_id",
            "query",
            "reference_doc_ids",
            "reference_relevant_context",
            "reference_answer",
        }
    elif case_type == 'case2':
        dataset_file = data_path / "case2_query_doc_dataset.csv"
        required_columns = {"query_id", "query", "reference_doc_ids"}
    else:
        raise ValueError(f"Unsupported case type: {case_type}")

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    logger.info(f"Loading data from {dataset_file}")
    df = pd.read_csv(dataset_file)
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset {dataset_file.name} missing columns: {sorted(missing_columns)}")

    df = df.fillna("")

    queries = df.to_dict(orient='records')
    if not queries:
        raise ValueError(f"Dataset {dataset_file.name} is empty.")

    if max_queries is not None and max_queries > 0 and len(queries) > max_queries:
        rng = random.Random(seed)
        sampled = list(queries)
        rng.shuffle(sampled)
        queries = sampled[:max_queries]
        logger.info(
            "Using query-limited dataset: %d/%d queries (seed=%d).",
            len(queries),
            len(df),
            seed,
        )

    return RAGDataset(queries, corpus, case_type)