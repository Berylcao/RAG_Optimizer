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

## How the RAG Optimizer Works

| Component | Library | Role |
|-----------|---------|------|
| Optimization | **Optuna** | TPE Bayesian sampler + MedianPruner for efficient search and early stopping |
| LLM Orchestration | **DSPy** | Chain-of-thought generation, query rewriting, and judge-style evaluation |
| Embeddings & Reranking | **sentence-transformers** | Local embedding models and cross-encoder rerankers |
| Sparse Retrieval | **rank_bm25** | BM25Okapi for keyword-based retrieval |
| API Calls | **OpenAI SDK** | OpenAI-compatible calls to teacher/student models |


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

## Sample Data

The project uses **HotpotQA** (multi-hop QA over Wikipedia). Sample data is provided under `data/hotpotqa/`:

- `case1_eval_dataset.csv` — queries with reference context and reference answers
- `case2_query_doc_dataset.csv` — queries with reference doc IDs only
- `reference_corpus.jsonl` — shared document corpus

To regenerate the dataset (requires internet):

```bash
python -m scripts.prepare_dataset
```

## How to Run on the Provided Sample Data

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS; on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the example and edit `.env` with your OpenAI-compatible endpoint(s):

```bash
cp example.env .env
```

### 3. Download local models (for dense/hybrid retrieval and reranking)

```bash
python -m src.utils.model_download_cli
```

### 4. Run the optimizer

**Case 1** (strong supervision; uses gold context and gold answers):

```bash
python main.py --case-type case1 --n-trials 6 --output-root outputs
```

**Case 2** (weak supervision; uses LLM-as-judge for groundedness and citation quality):

```bash
python main.py --case-type case2 --n-trials 6 --output-root outputs
```

By default, data is read from `data/hotpotqa/`. Override with `--data-dir`:

```bash
python main.py --case-type case1 --n-trials 6 --data-dir /path/to/your/data --output-root outputs
```

Your data directory must contain: `case1_eval_dataset.csv`, `case2_query_doc_dataset.csv`, and `reference_corpus.jsonl` (schema as in the sample).

### 5. Outputs and recommendation report

Results are written to `outputs/case1/` and `outputs/case2/`. Each case folder contains:

| File | Description |
|------|-------------|
| `best_config.json` | Selected RAG pipeline configuration |
| `README.md` | **Recommendation report** — explains why this config was selected and documents tuning/holdout scores |
| `run_summary.csv` | All trials: score, latency, status, config |
| `per_query_diagnostics.csv` | Per-query metrics, error type, retrieved docs, generated answer |
| `retrieval_examples/`, `answer_examples/` | Per-query JSON files for inspection |

The `README.md` in each output folder serves as the **output report** that recommends the selected configuration for that case and explains the selection logic.

**Note**: Current results were obtained using Qwen3.5 Flash as the teacher model and a locally deployed Qwen2.5-1.5b (via Ollama) as the student model.

## Configuration

- **Search space**: `configs/search_space.yaml` — defines tunable parameters (chunking, indexing, reranking, query refinement, generation).
- **Evaluation settings**: `configs/evaluation_settings.yaml` — metric weights, holdout fraction, top-k.
- **Model paths**: `configs/model_paths.yaml` — local embedding and reranker models.

## Project Structure

```
rag_optimizer/
├── main.py                     # Entry point
├── configs/                    # Search space and evaluation config
├── scripts/
│   ├── run_optimization.py     # Core optimization loop
│   └── prepare_dataset.py     # HotpotQA data preparation
├── src/
│   ├── core/                   # Caching, experiment tracking
│   ├── evaluation/             # Evaluators and metrics
│   ├── pipelines/              # Chunker, retriever, reranker, generator
│   └── utils/                  # Data loader, model loader, model download CLI
├── data/hotpotqa/              # Sample dataset
├── outputs/                    # Experiment outputs (best_config, run_summary, etc.)
├── example.env                 # Template for .env configuration
├── requirements.txt
└── pyproject.toml
```
