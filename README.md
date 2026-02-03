# Deep Thinking RAG

An advanced Retrieval-Augmented Generation pipeline that uses agentic reasoning to handle complex, multi-hop queries. Unlike traditional linear RAG systems, this implementation uses a cyclical workflow with planning, adaptive retrieval, and self-critique.

## Features

- **Multi-step Planning**: Decomposes complex queries into structured research plans
- **Adaptive Retrieval**: Dynamically selects between vector, keyword, and hybrid search
- **Multi-source Knowledge**: Combines internal documents, web search, and stock metrics
- **Cross-encoder Reranking**: High-precision document filtering
- **Self-critique Loop**: Policy agent decides when to continue or finish research
- **Citation Support**: Final answers include source references

## Architecture

```
User Query
    │
    ▼
┌─────────┐     ┌─────────────┐     ┌─────────┐
│  Plan   │────▶│  Retrieve   │────▶│ Rerank  │
└─────────┘     │ (10K/Web/   │     └────┬────┘
                │  Metrics)   │          │
                └─────────────┘          ▼
                      ▲            ┌──────────┐
                      │            │ Compress │
                ┌─────┴─────┐      └────┬─────┘
                │  Continue │           │
                └─────┬─────┘           ▼
                      │           ┌──────────┐
                      ◀───────────│ Reflect  │
                                  └────┬─────┘
                ┌─────────┐            │
                │ Finish  │◀───────────┘
                └────┬────┘
                     ▼
              ┌────────────┐
              │   Final    │
              │   Answer   │
              └────────────┘
```

## Project Structure

```
deep-thinking-rag/
├── src/
│   ├── config/          # Configuration settings
│   ├── data/            # Document loading & processing
│   ├── retrieval/       # Vector, BM25, hybrid search, reranking
│   ├── agents/          # Planner, supervisor, distiller, policy
│   ├── graph/           # LangGraph state and workflow
│   ├── pipelines/       # Shallow and deep RAG implementations
│   └── evaluation/      # RAGAs evaluation metrics
├── data/
│   ├── raw/             # Raw source documents
│   └── processed/       # Processed/chunked documents
├── notebooks/           # Original Jupyter notebook (reference)
├── run.py               # Main CLI entry point
├── requirements.txt
└── .env.example
```

## Installation

**Requirements:** Python 3.11 or 3.12 (Python 3.13 has compatibility issues)

```bash
# Clone the repository
git clone https://github.com/yourusername/deep-thinking-rag.git
cd deep-thinking-rag

# Create virtual environment
python -m venv venv

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (Windows PowerShell - if allowed)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Add your API keys to `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   FINNHUB_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here  # Optional, for tracing
   ```

## Usage

### Run with default query
```bash
python run.py
```

### Run with custom query
```bash
python run.py --query "Your complex question here"
```

### Compare with baseline RAG
```bash
python run.py --baseline
```

### Use in Python
```python
from src.pipelines.deep_rag import run_deep_thinking_rag, display_final_answer

# Run the pipeline
final_state = run_deep_thinking_rag(
    "Based on NVIDIA's 2023 10-K filing, identify their key risks..."
)

# Display the answer
display_final_answer(final_state)

# Get raw answer text
answer = final_state['final_answer']
```

## Default Test Query

The default query demonstrates multi-hop reasoning across internal documents and web search:

> "Based on NVIDIA's 2023 10-K filing, identify their key risks related to competition. Then, find recent news about AMD's AI chip strategy and explain how this strategy addresses or exacerbates one of NVIDIA's stated risks."

## Key Components

| Component | Purpose |
|-----------|---------|
| **Planner Agent** | Decomposes queries into sub-questions with tool selection |
| **Retrieval Supervisor** | Chooses optimal search strategy per sub-question |
| **Query Rewriter** | Optimizes queries for better retrieval |
| **Cross-Encoder** | Reranks documents for precision |
| **Distiller Agent** | Compresses context into concise summaries |
| **Policy Agent** | Decides to continue research or finish |

## Knowledge Sources

1. **10-K Documents**: NVIDIA's 2023 annual filing (included)
2. **Web Search**: Real-time information via Tavily API
3. **Stock Metrics**: Fundamentals and analyst data via Finnhub API

## License

MIT
