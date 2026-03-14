# Output Summary

## Case
`case2`

## How Evaluation Works in This Case (No Reference Answer)
Case 2 has **no gold answer or reference relevant context**. Evaluation therefore uses:
- **retrieval_coverage_proxy**: fraction of reference doc IDs that appear in retrieved doc IDs (retrieval-centric).
- **groundedness**: LLM-as-judge (TEACHER) scores how well the answer is grounded in the retrieved context (groundedness-oriented).
- **citation_quality**: LLM-as-judge scores citation correctness against retrieved doc IDs.
The overall score is a weighted sum of these three. **This is how evaluation changes when no reference answer exists**: we optimize for retrieval coverage and model-based groundedness/citation instead of similarity to a gold answer.

## Recommended Configuration
```json
{
  "chunking": {
    "size": 128,
    "overlap": 32,
    "strategy": "semantic"
  },
  "indexing": {
    "retriever": "hybrid",
    "embedding_model": "e5-base",
    "metadata_enrichment": true
  },
  "reranking": {
    "enabled": true,
    "model": "bge-reranker-base"
  },
  "query_refinement": {
    "rewrite": false,
    "decompose": false
  },
  "generation": {
    "llm": "student-direct",
    "temperature": 0.2,
    "answer_style": "citation_first"
  }
}
```

## Selection Details
- `selection_rule`: overall score -> case-specific quality metrics -> lower latency -> lower trial number
- `selected_trial_number`: 1
- `selected_score`: 0.72
- `latency_seconds`: 1171.7589
- `groundedness`: 0.6111
- `citation_quality`: 0.6556
- `retrieval_coverage_proxy`: 0.8333
- `tuning_score`: 0.72
- `holdout_score`: 0.8333
- `holdout_retrieval_coverage_proxy`: 0.9167
- `holdout_groundedness`: 0.7833
- `holdout_citation_quality`: 0.7333

## Aggregate Metrics
- `retrieval_coverage_proxy`: 0.8333
- `groundedness`: 0.6111
- `citation_quality`: 0.6556

## Overall Score
- `score`: 0.7200

## Latency
- `latency_seconds`: 1171.7589
