# Output Summary

## Case
`case1`

## How Evaluation Works in This Case (With Reference Context and Answer)
Case 1 has **reference relevant context** and **reference answer**. Evaluation uses:
- **context_recall**: retrieval alignment — fraction of reference-context tokens covered by retrieved context.
- **answer_similarity**: generated vs reference answer (semantic + token overlap).
- **faithfulness**: fraction of answer tokens present in retrieved context.
Overall score is a weighted sum. Optimization targets both retrieval quality and answer quality against the gold reference.

## Recommended Configuration
```json
{
  "chunking": {
    "size": 512,
    "overlap": 64,
    "strategy": "token"
  },
  "indexing": {
    "retriever": "hybrid",
    "embedding_model": "e5-base",
    "metadata_enrichment": false
  },
  "reranking": {
    "enabled": true,
    "model": "bge-reranker-base"
  },
  "query_refinement": {
    "rewrite": true,
    "decompose": true
  },
  "generation": {
    "llm": "gpt-class",
    "temperature": 0.0,
    "answer_style": "concise"
  }
}
```

## Selection Details
- `selection_rule`: overall score -> case-specific quality metrics -> lower latency -> lower trial number
- `selected_trial_number`: 4
- `selected_score`: 0.8203
- `latency_seconds`: 692.2013
- `context_recall`: 0.9702
- `answer_similarity`: 0.6047
- `faithfulness`: 0.8605
- `tuning_score`: 0.8203
- `holdout_score`: 0.8518
- `holdout_context_recall`: 0.9583
- `holdout_answer_similarity`: 0.7085
- `holdout_faithfulness`: 0.8629

## Aggregate Metrics
- `context_recall`: 0.9702
- `answer_similarity`: 0.6047
- `faithfulness`: 0.8605

## Overall Score
- `score`: 0.8203

## Latency
- `latency_seconds`: 692.2013
