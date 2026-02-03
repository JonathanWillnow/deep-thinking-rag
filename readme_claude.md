# Deep Thinking RAG - Project Structure Guide

This document explains the modular folder structure designed to make this RAG codebase maintainable, extensible, and easy for coding agents to navigate.

## Directory Overview

```
deep-thinking-rag/
├── src/                    # All source code
│   ├── config/             # Configuration management
│   ├── data/               # Data loading and processing
│   ├── retrieval/          # Search and retrieval components
│   ├── agents/             # LLM-powered agents
│   ├── graph/              # LangGraph workflow orchestration
│   ├── pipelines/          # End-to-end RAG pipelines
│   └── evaluation/         # Metrics and evaluation
├── data/                   # Data storage
│   ├── raw/                # Original unprocessed files
│   └── processed/          # Cleaned/chunked data
├── notebooks/              # Jupyter notebooks for experimentation
├── tests/                  # Unit and integration tests
├── run.py                  # Main CLI entry point
├── .env.example            # Environment variables template
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Module Descriptions

### `src/config/`
**Purpose:** Centralized configuration for the entire application.

| File | Responsibility |
|------|----------------|
| `settings.py` | All configuration constants (model names, chunk sizes, API keys via env vars) |

**When to modify:** When adding new configurable parameters, changing model names, or adjusting hyperparameters.

---

### `src/data/`
**Purpose:** Everything related to data ingestion, cleaning, chunking, and embedding generation.

| File | Responsibility |
|------|----------------|
| `loader.py` | Load documents from various sources (HTML, PDF, text files) |
| `processor.py` | Clean text, chunk documents with metadata preservation |
| `embeddings.py` | Generate and manage vector embeddings |

**When to modify:** When adding new document types, changing chunking strategies, or switching embedding models.

---

### `src/retrieval/`
**Purpose:** All search and retrieval mechanisms.

| File | Responsibility |
|------|----------------|
| `vector_store.py` | ChromaDB operations (add, query, persist) |
| `bm25.py` | Keyword-based BM25 search implementation |
| `hybrid.py` | Hybrid search with Reciprocal Rank Fusion (RRF) |
| `reranker.py` | Cross-encoder reranking for precision |

**When to modify:** When adding new retrieval strategies, tuning search parameters, or integrating new vector databases.

---

### `src/agents/`
**Purpose:** LLM-powered agents that perform specialized tasks.

| File | Responsibility |
|------|----------------|
| `planner.py` | Decomposes complex queries into research steps |
| `retrieval_supervisor.py` | Selects optimal retrieval strategy (vector/BM25/hybrid) |
| `query_rewriter.py` | Optimizes queries for better retrieval |
| `distiller.py` | Compresses retrieved context into concise evidence |
| `policy.py` | Decides when to continue research vs. finish |

**When to modify:** When adding new agent capabilities, changing prompts, or adjusting agent behavior.

---

### `src/graph/`
**Purpose:** LangGraph workflow orchestration.

| File | Responsibility |
|------|----------------|
| `state.py` | RAGState TypedDict definition |
| `nodes.py` | Individual graph node functions |
| `workflow.py` | StateGraph construction and compilation |

**When to modify:** When changing the workflow logic, adding new nodes, or modifying routing conditions.

---

### `src/pipelines/`
**Purpose:** Complete end-to-end RAG implementations.

| File | Responsibility |
|------|----------------|
| `shallow_rag.py` | Baseline linear RAG pipeline |
| `deep_rag.py` | Advanced agentic RAG pipeline |

**When to modify:** When creating new pipeline variants or modifying the overall flow.

---

### `src/evaluation/`
**Purpose:** Metrics and evaluation framework.

| File | Responsibility |
|------|----------------|
| `metrics.py` | RAGAS evaluation (faithfulness, relevance, precision, recall) |

**When to modify:** When adding new evaluation metrics or changing evaluation methodology.

---

### `data/`
**Purpose:** Data storage separated from code.

- `raw/` - Original source files (e.g., `nvda_10k_2023_raw.html`)
- `processed/` - Cleaned and chunked data (e.g., `nvda_10k_2023_clean.txt`)

---

### `notebooks/`
**Purpose:** Jupyter notebooks for experimentation and prototyping.

The original `deepRAG.ipynb` should be moved here. Notebooks are for exploration; production code goes in `src/`.

---

### `tests/`
**Purpose:** Test coverage for all modules.

Mirror the `src/` structure:
```
tests/
├── test_config/
├── test_data/
├── test_retrieval/
├── test_agents/
├── test_graph/
├── test_pipelines/
└── test_evaluation/
```

---

### `run.py` (Root)
**Purpose:** Main CLI entry point for the RAG pipeline.

| File | Responsibility |
|------|----------------|
| `run.py` | Main CLI to run RAG queries (`python run.py --query "..."`) |

---

## Design Principles

### 1. Single Responsibility
Each file has one clear purpose. This makes it easy to locate code and understand its role.

### 2. Dependency Flow
```
config → data → retrieval → agents → graph → pipelines → evaluation
```
Dependencies flow left-to-right. Lower-level modules don't import from higher-level ones.

### 3. Configuration Externalization
All magic numbers, model names, and parameters live in `src/config/settings.py`. This enables easy experimentation without code changes.

### 4. Separation of Concerns
- **Data layer** (`src/data/`): How documents are loaded and processed
- **Retrieval layer** (`src/retrieval/`): How information is found
- **Agent layer** (`src/agents/`): How decisions are made
- **Orchestration layer** (`src/graph/`): How components connect
- **Pipeline layer** (`src/pipelines/`): How everything runs end-to-end

### 5. Testability
Modular design enables unit testing of individual components in isolation.

---

## Quick Navigation Guide for Agents

| Task | Look Here |
|------|-----------|
| Change model or parameters | `src/config/settings.py` |
| Modify document processing | `src/data/processor.py` |
| Tune retrieval | `src/retrieval/` |
| Change agent prompts | `src/agents/` |
| Modify workflow logic | `src/graph/nodes.py`, `src/graph/workflow.py` |
| Add evaluation metrics | `src/evaluation/metrics.py` |
| Run experiments | `notebooks/` |
| Run production queries | `run.py` |

---

## Extension Points

To extend this system:

1. **New retrieval strategy**: Add a new file in `src/retrieval/`, implement the search interface
2. **New agent**: Add a new file in `src/agents/`, follow the existing pattern
3. **New data source**: Extend `src/data/loader.py` with a new loader function
4. **New evaluation metric**: Add to `src/evaluation/metrics.py`
5. **New pipeline variant**: Create a new file in `src/pipelines/`
