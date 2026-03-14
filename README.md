# Automated RAG Pipeline Optimizer

This is an end-to-end tool that automatically searches over chunking, indexing, reranking, query refinement, and generation strategies to recommend an optimal RAG pipeline configuration for a given dataset.

## Highlight Features

- **Full-pipeline optimization** — searches across five RAG dimensions simultaneously: chunking (size, overlap, strategy), indexing (BM25, dense, hybrid), reranking (cross-encoder models), query refinement (LLM rewriting, decomposition), and generation (model, temperature, answer style).
- **Two evaluation modes** — Case 1 (strong supervision with gold answers) uses context recall, answer similarity, and faithfulness; Case 2 (no gold answer) uses retrieval coverage, LLM-as-judge groundedness, and citation quality.
- **Bayesian optimization with early stopping** — Optuna TPE sampler explores the search space efficiently, while MedianPruner prunes underperforming trials mid-run.
- **Tuning / holdout split** — 60% of queries for optimizer tuning, 40% held out for generalization check, preventing overfitting to small benchmarks.
- **LLM-as-judge evaluation** — Case 2 uses a teacher model (configurable via `.env`) to score groundedness and citation quality via DSPy-powered chain-of-thought judging.
- **Deterministic and reproducible** — fixed random seeds, config-driven pipeline construction, structured experiment logs (`trial_events.jsonl`, `failed_trials.jsonl`), and per-query diagnostics.
- **Multi-level caching** — in-memory model caching, per-run index reuse for identical chunking+indexing configs, and disk-based chunk caching.
- **Explicit tie-breaking** — overall score → case-specific quality metrics → lower latency → lower trial number. Selection rule is documented in every output report.

## Dataset

This project uses **HotpotQA** (multi-hop question answering over Wikipedia) as the open-source evaluation dataset, stored under `data/hotpotqa/`:

To regenerate the dataset (requires internet):

```bash
python -m scripts.prepare_dataset [--n-case1 15] [--n-case2 15] [--seed 42]
```

## Models

### LLM Models (Teacher & Student)

Configured entirely via `.env`. Any **OpenAI-compatible** endpoint works (OpenAI, Azure, DashScope, Ollama, etc.):

| Role | Used for | Configuration |
|------|----------|---------------|
| **Teacher** | DSPy query rewriting, Case 2 judge-style evaluation (groundedness, citation quality) | `TEACHER_MODEL_NAME`, `TEACHER_API_KEY`, `TEACHER_API_BASE` |
| **Student** | Answer generation in the RAG pipeline | `STUDENT_MODEL_NAME`, `STUDENT_API_KEY`, `STUDENT_API_BASE` |

### Local Embedding & Reranking Models

Downloaded via the built-in model download utility (`src/utils/model_download_cli.py`):

| Model | Type | Used for |
|-------|------|----------|
| `bge-small-en-v1.5` | Embedding | Dense retrieval, hybrid retrieval, semantic chunking |
| `e5-base-v2` | Embedding | Dense retrieval, semantic similarity metrics |
| `bge-reranker-base` | Cross-encoder | Reranking retrieved passages |
| `ms-marco-MiniLM-L-6-v2` | Cross-encoder | Reranking retrieved passages |

Download all models:

```bash
python -m src.utils.model_download_cli
```

## How the RAG Optimizer Works

### Tech Stack

| Component | Library | Role |
|-----------|---------|------|
| Optimization | **Optuna** | TPE Bayesian sampler + MedianPruner for efficient search and early stopping |
| LLM Orchestration | **DSPy** | Chain-of-thought generation, query rewriting, and judge-style evaluation |
| Embeddings & Reranking | **sentence-transformers** | Local embedding models and cross-encoder rerankers |
| Sparse Retrieval | **rank_bm25** | BM25Okapi for keyword-based retrieval |
| API Calls | **OpenAI SDK** | OpenAI-compatible calls to teacher/student models |


## Output Structure

Each run writes to `outputs/{case_type}/`:

```
outputs/
├── case1/
│   ├── best_config.json              # Winning RAG pipeline configuration
│   ├── run_summary.csv               # All trials: score, latency, status, config
│   ├── per_query_diagnostics.csv     # Per-query: retrieval/generation scores, error type, retrieved docs, answer
│   ├── retrieval_examples/
│   │   └── {query_id}.json           # Retrieved doc IDs, chunks, metrics per query
│   ├── answer_examples/
│   │   └── {query_id}.json           # Generated answer, reference answer, error type per query
│   ├── param_importance.json         # Optuna hyperparameter importance ranking
│   ├── trial_events.jsonl            # Structured event log for experiment traceability
│   ├── failed_trials.jsonl           # Failed trial details with traceback
│   └── README.md                     # Recommendation report: why this config won
├── case2/
│   └── (same structure)
└── README.md                         # Output directory overview
```
Note: Current results were obtained using Qwen3.5 Flash as the teacher model and a locally deployed Qwen2.5-1.5b (via Ollama) as the student model.

## Getting Started

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd rag_optimizer
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the example and fill in your keys:

```bash
cp example.env .env
```

Edit `.env` with your model endpoints. Any OpenAI-compatible API works:

For local Ollama:

```env
STUDENT_MODEL_NAME="llama3"
STUDENT_API_KEY=""
STUDENT_API_BASE="http://localhost:11434"
```

### 3. Download local models

Embedding and reranking models are required for dense/hybrid retrieval and reranking:

```bash
python -m src.utils.model_download_cli
```

### 4. Prepare dataset (optional — data is included)

Data is already provided under `data/hotpotqa/`. To regenerate:

```bash
python -m scripts.prepare_dataset
```

### 5. Run the optimizer

```bash
# Case 1: strong supervision (gold context + gold answer)
python main.py --case-type case1 --n-trials 8 --output-root outputs

# Case 2: weak supervision (no gold answer, uses LLM-as-judge)
python main.py --case-type case2 --n-trials 8 --output-root outputs
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--case-type` | (required) | `case1` or `case2` |
| `--n-trials` | 8 | Number of pipeline configurations to try |
| `--output-root` | `outputs` | Where to write results |
| `--data-dir` | `data/hotpotqa` | Path to dataset |
| `--search-space` | `configs/search_space.yaml` | Search space definition |
| `--evaluation-settings` | `configs/evaluation_settings.yaml` | Evaluation weights and settings |
| `--max-queries` | all | Limit dataset to N queries for faster runs |

## Project Structure

```
rag_optimizer/
├── main.py                          # Entry point
├── configs/
│   ├── search_space.yaml            # Search space definition
│   ├── evaluation_settings.yaml     # Metric weights, judge settings, holdout fraction
│   └── model_paths.yaml             # Local model registry
├── scripts/
│   ├── run_optimization.py          # Core optimization loop
│   └── prepare_dataset.py           # HotpotQA data preparation
├── src/
│   ├── core/
│   │   ├── cache_manager.py         # Disk-based chunk caching
│   │   └── experiment_tracker.py    # Output writing: CSV, JSON, README reports
│   ├── evaluation/
│   │   ├── base.py                  # BaseMetric, BaseEvaluator, EvaluationResult
│   │   ├── evaluators.py            # Case1Evaluator, Case2Evaluator
│   │   └── metrics.py               # All metric implementations
│   ├── pipelines/
│   │   ├── base.py                  # Base classes: Chunker, Retriever, Reranker, Generator
│   │   ├── components.py            # All pipeline component implementations
│   │   ├── dspy_modules.py          # DSPy signatures and modules
│   │   └── factory.py               # PipelineFactory: config → pipeline
│   └── utils/
│       ├── data_loader.py           # Dataset loading and splitting
│       ├── dspy_model_loader.py     # .env → model config (OpenAI-compatible)
│       ├── local_model_registry.py  # Local embedding/reranker model loading
│       ├── hf_model_downloader.py   # HuggingFace model download logic
│       └── model_download_cli.py    # CLI for downloading local models
├── tests/
│   ├── test_formal_mode_guards.py   # Strict mode guard tests
│   └── test_selection_and_tracking.py  # Tie-breaking and tracker tests
├── data/hotpotqa/                   # HotpotQA evaluation dataset
├── outputs/                         # Experiment outputs (per case)
├── example.env                      # Template for .env configuration
├── requirements.txt
└── pyproject.toml
```
