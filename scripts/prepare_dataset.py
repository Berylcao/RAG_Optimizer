"""
Download a subset of HotpotQA and convert it to the project's data format.

HotpotQA is a multi-hop QA dataset where each question requires reasoning
over 2+ Wikipedia paragraphs.  It provides gold answers, supporting facts,
and distractor paragraphs — a natural fit for RAG evaluation.

Usage:
    python -m scripts.prepare_dataset [--n-case1 15] [--n-case2 15] [--seed 42]
"""

import argparse
import csv
import json
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "hotpotqa"


def _fetch_hotpotqa(split: str = "validation"):
    from datasets import load_dataset

    logger.info("Downloading HotpotQA (distractor, %s) …", split)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    logger.info("HotpotQA: %d examples available.", len(ds))
    return ds


def _build_corpus_and_queries(
    examples: list,
    id_prefix: str,
) -> tuple[list[dict], list[dict]]:
    """Convert raw HotpotQA examples into (corpus_docs, query_records)."""
    corpus: dict[str, dict] = {}
    queries: list[dict] = []

    for idx, ex in enumerate(examples, start=1):
        titles = ex["context"]["title"]
        sentences_list = ex["context"]["sentences"]
        sf_titles = set(ex["supporting_facts"]["title"])

        ref_doc_ids: list[str] = []
        ref_context_parts: list[str] = []

        for title, sents in zip(titles, sentences_list):
            doc_id = f"D{abs(hash(title)) % 100000:05d}"
            text = " ".join(sents).strip()
            if not text:
                continue
            if doc_id not in corpus:
                corpus[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "source": "hotpotqa_wikipedia",
                }
            if title in sf_titles:
                ref_doc_ids.append(doc_id)
                sf_indices = [
                    i for t, i in zip(
                        ex["supporting_facts"]["title"],
                        ex["supporting_facts"]["sent_id"],
                    ) if t == title
                ]
                for si in sf_indices:
                    if si < len(sents):
                        ref_context_parts.append(sents[si].strip())

        queries.append({
            "query_id": f"{id_prefix}{idx:03d}",
            "query": ex["question"],
            "reference_doc_ids": ref_doc_ids,
            "reference_relevant_context": " ".join(ref_context_parts),
            "reference_answer": ex["answer"],
        })

    return list(corpus.values()), queries


def _write_corpus(docs: list[dict], out_dir: Path) -> None:
    path = out_dir / "reference_corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info("Wrote %d documents to %s", len(docs), path)


def _write_case1(queries: list[dict], out_dir: Path) -> None:
    path = out_dir / "case1_eval_dataset.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            "query_id", "query", "reference_doc_ids",
            "reference_relevant_context", "reference_answer",
        ])
        for q in queries:
            writer.writerow([
                q["query_id"],
                q["query"],
                "|".join(q["reference_doc_ids"]),
                q["reference_relevant_context"],
                q["reference_answer"],
            ])
    logger.info("Wrote %d Case 1 queries to %s", len(queries), path)


def _write_case2(queries: list[dict], out_dir: Path) -> None:
    """Write Case 2 CSV — only query + reference_doc_ids, no gold answer."""
    path = out_dir / "case2_query_doc_dataset.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["query_id", "query", "reference_doc_ids"])
        for q in queries:
            writer.writerow([
                q["query_id"],
                q["query"],
                "|".join(q["reference_doc_ids"]),
            ])
    logger.info("Wrote %d Case 2 queries to %s", len(queries), path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare HotpotQA data for RAG optimizer evaluation.",
    )
    parser.add_argument("--n-case1", type=int, default=15, help="Number of Case 1 queries (default 15).")
    parser.add_argument("--n-case2", type=int, default=15, help="Number of Case 2 queries (default 15).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.n_case1 + args.n_case2
    ds = _fetch_hotpotqa("validation")

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = [ds[i] for i in indices[:total]]

    case1_raw = selected[: args.n_case1]
    case2_raw = selected[args.n_case1:]

    c1_corpus, case1_queries = _build_corpus_and_queries(case1_raw, id_prefix="C1Q")
    c2_corpus, case2_queries = _build_corpus_and_queries(case2_raw, id_prefix="C2Q")

    all_corpus: dict[str, dict] = {}
    for d in c1_corpus:
        all_corpus[d["doc_id"]] = d
    for d in c2_corpus:
        all_corpus.setdefault(d["doc_id"], d)
    corpus_final = list(all_corpus.values())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_corpus(corpus_final, OUTPUT_DIR)
    _write_case1(case1_queries, OUTPUT_DIR)
    _write_case2(case2_queries, OUTPUT_DIR)

    logger.info(
        "Done. Corpus: %d docs | Case 1: %d queries | Case 2: %d queries",
        len(corpus_final), len(case1_queries), len(case2_queries),
    )
    logger.info("Output directory: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
