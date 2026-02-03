# Deep Thinking RAG - Migration Plan

## Overview
This document outlines the migration of code from `deepRAG.ipynb` to the modular folder structure.

## Code Mapping

### 1. Configuration (`src/config/settings.py`)
**Source:** Lines 344-363 of notebook
- Global `CONFIG` dictionary
- Environment variable setup
- LangSmith tracing configuration
- Directory paths

### 2. Data Layer (`src/data/`)

#### `loader.py`
- `download_and_parse_10k()` - Download and parse SEC 10-K filings
- Document loading utilities

#### `processor.py`
- Text splitting with `RecursiveCharacterTextSplitter`
- Section extraction with metadata
- `doc_chunks_with_metadata` creation

#### `embeddings.py`
- OpenAI embeddings setup

### 3. Retrieval Layer (`src/retrieval/`)

#### `vector_store.py`
- ChromaDB setup (baseline and advanced)
- `vector_search_only()` function

#### `bm25.py`
- BM25Okapi index setup
- `bm25_search_only()` function

#### `hybrid.py`
- `hybrid_search()` with Reciprocal Rank Fusion

#### `reranker.py`
- CrossEncoder initialization
- `rerank_documents_function()`

### 4. Agents Layer (`src/agents/`)

#### `planner.py`
- `Step` Pydantic model
- `Plan` Pydantic model
- `planner_agent` with tool-aware prompt

#### `retrieval_supervisor.py`
- `RetrievalDecision` model
- `retrieval_supervisor_agent`

#### `query_rewriter.py`
- `query_rewriter_agent`

#### `distiller.py`
- `distiller_agent` for context compression

#### `policy.py`
- `Decision` model
- `policy_agent` (LLM-as-a-Judge)
- `reflection_agent`

#### `web_search.py`
- Tavily search tool setup
- `web_search_function()`

#### `metrics_fetcher.py`
- Finnhub API functions
- `fetch_stock_rag_documents()`
- `fetch_fundamentals()`, `fetch_analyst_opinions()`

### 5. Graph Layer (`src/graph/`)

#### `state.py`
- `RAGState` TypedDict
- `PastStep` TypedDict

#### `nodes.py`
- `plan_node()`
- `retrieval_node()`
- `web_search_node()`
- `retrieve_metrics_node()`
- `rerank_node()`
- `compression_node()`
- `reflection_node()`
- `final_answer_node()`
- Helper: `get_past_context_str()`
- Helper: `format_docs()`

#### `workflow.py`
- `route_by_tool()` conditional edge
- `should_continue_node()` conditional edge
- StateGraph construction
- Graph compilation

### 6. Pipelines Layer (`src/pipelines/`)

#### `shallow_rag.py`
- Baseline RAG chain (LCEL)

#### `deep_rag.py`
- Deep Thinking RAG graph instantiation

### 7. Evaluation Layer (`src/evaluation/`)

#### `metrics.py`
- RAGAs evaluation setup
- Evaluation dataset construction

### 8. Entry Point (`scripts/run.py`)
- CLI interface to run the pipeline
- Argument parsing for query input

## Execution Order
1. config (no dependencies)
2. data (depends on config)
3. retrieval (depends on config, data)
4. agents (depends on config)
5. graph (depends on all above)
6. pipelines (depends on graph)
7. evaluation (depends on pipelines)
8. scripts/run.py (depends on pipelines)
