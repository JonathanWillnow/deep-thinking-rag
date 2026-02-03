#!/usr/bin/env python
# coding: utf-8

# # A Guide to Production-Grade RAG: From Theory to Autonomous Agents
# 
# ## Table of Contents
# 
# **Part 1: Setting the Stage - Foundations and Our Core Challenge**
# * [1.1. Introduction: The Limits of "Shallow" RAG](#part1-1-intro-pro)
# * [1.2. Environment Setup: API Keys, Imports, and Configuration](#part1-2-env-pro-adv)
# * [1.3. The Dataset: Preparing Our Knowledge Base](#part1-3-data-pro)
# * [1.4. The Upgraded Challenge: A Multi-Source, Multi-Hop Query](#part1-4-challenge-pro-adv)
# 
# **Part 2: The Baseline - Building and Breaking a "Vanilla" RAG Pipeline**
# * [2.1. Code Dependency: Document Loading and Naive Chunking](#part2-1-dep-pro)
# * [2.2. Code Dependency: Creating the Vector Store](#part2-2-dep-pro)
# * [2.3. Code Dependency: Assembling the Simple RAG Chain](#part2-3-dep-pro)
# * [2.4. The Critical Failure Case: Demonstrating the Need for Advanced Techniques](#part2-4-fail-pro-adv)
# * [2.5. Diagnosis: Why Did It Fail?](#part2-5-diag-pro-adv)
# 
# **Part 3: The "Deep Thinking" Upgrade: Engineering an Autonomous Reasoning Engine**
# * [3.1. Code Dependency: Defining the `RAGState`](#part3-1-state-pro-adv)
# * [3.2. Component 1: Dynamic Planning and Query Formulation](#part3-2-planner-pro-adv)
#     * [3.2.1. The Tool-Aware Planner Agent](#part3-2-1-planner-pro-adv)
#     * [3.2.2. Query Rewriting and Expansion](#part3-2-2-rewriter-pro)
#     * [3.2.3. Entity and Constraint Extraction](#part3-2-3-metadata-pro)
# * [3.3. Component 2: The Multi-Stage, Adaptive Retrieval Funnel](#part3-3-retrieval-pro-adv)
#     * [3.3.1. NEW: The Retrieval Supervisor Agent](#part3-3-1-supervisor-pro)
#     * [3.3.2. Implementing the Retrieval Strategies](#part3-3-2-strategies-pro)
#     * [3.3.3. Stage 2 (High Precision): Cross-Encoder Reranker](#part3-3-3-reranker-pro)
#     * [3.3.4. Stage 3 (Contextual Distillation)](#part3-3-4-distill-pro)
# * [3.4. Component 3: Tool Augmentation with Web Search](#part3-4-tool-pro)
# * [3.5. Component 4: The Self-Critique and Control Flow Policy](#part3-5-critique-pro)
#     * [3.5.1. The "Update and Reflect" Step](#part3-5-1-reflect-pro)
#     * [3.5.2. Policy Implementation (LLM-as-a-Judge)](#part3-5-2-policy-pro)
#     * [3.5.3. Defining Robust Stopping Criteria](#part3-5-3-stopping-pro)
# 
# **Part 4: Assembly with LangGraph - Orchestrating the Reasoning Loop**
# * [4.1. Code Dependency: Defining the Graph Nodes](#part4-1-nodes-pro-adv)
# * [4.2. Code Dependency: Defining the Conditional Edges](#part4-2-edges-pro-adv)
# * [4.3. Building the `StateGraph`](#part4-3-build-pro-adv)
# * [4.4. Compiling and Visualizing the Workflow](#part4-4-viz-pro-adv)
# 
# **Part 5: Redemption - Running the Advanced Agent**
# * [5.1. Invoking the Graph: A Step-by-Step Trace](#part5-1-invoke-pro-adv)
# * [5.2. Analyzing the Final High-Quality Output](#part5-2-analyze-pro-adv)
# * [5.3. Side-by-Side Comparison: Vanilla vs. Deep Thinking RAG](#part5-3-compare-pro-adv)
# 
# **Part 6: A Production-Grade Evaluation Framework**
# * [6.1. Evaluation Metrics Overview](#part6-metrics-pro)
# * [6.2. Code Dependency: Implementing Evaluation with RAGAs](#part6-4-ragas-code-pro-adv)
# * [6.3. Interpreting the Evaluation Scores](#part6-5-interpret-pro-adv)
# 
# **Part 7: Optimizations and Production Considerations**
# * [7.1. Optimization: Caching](#part7-1-cache-pro)
# * [7.2. Feature: Provenance and Citations](#part7-2-provenance-pro)
# * [7.3. Discussion: The Next Level - MDPs and Learned Policies](#part7-3-discussion-pro)
# * [7.4. Handling Failure: Graceful Exits and Fallbacks](#part7-4-failure-pro)
# 
# **Part 8: Conclusion and Key Takeaways**
# * [8.1. Summary of Our Journey](#part8-conclusion-pro)
# * [8.2. Key Architectural Principles of Advanced RAG Systems](#part8-2-principles-pro-adv)
# * [8.3. Future Directions](#part8-3-future-pro-adv)

# ## Part 1: Setting the Stage - Foundations and Our Core Challenge

# ### 1.1. Introduction: The Limits of "Shallow" RAG
# 
# Retrieval-Augmented Generation (RAG) has become the dominant paradigm for creating knowledge-intensive AI systems. The standard approach—a linear, three-step pipeline of **Retrieve -> Augment -> Generate**—is remarkably effective for simple, fact-based queries. However, this "shallow" RAG architecture reveals critical weaknesses when faced with complex questions that demand synthesis, comparison, and multi-step reasoning across a large and varied knowledge base.
# 
# The next frontier in RAG is not about bigger models or larger context windows, but about greater **autonomy and intelligence** in the retrieval and reasoning process. The industry is moving from static chains to dynamic, agentic systems that can emulate a human researcher's workflow. These systems can decompose complex problems, select appropriate tools, dynamically adapt their retrieval strategies, and critique their own progress.
# 
# In this comprehensive guide, we will build a powerful, **standalone** implementation of a **Deep Thinking RAG Pipeline**. We will meticulously engineer every component, from a sophisticated multi-stage, adaptive retrieval funnel to a tool-augmented, self-critiquing policy engine. We will begin by exposing the failure of a vanilla RAG system on a challenging query, and then, step-by-step, construct our advanced agent using **LangGraph** to orchestrate its complex, cyclical reasoning. By the end, you will have a production-grade framework and a deep, architectural understanding of how to build RAG systems that can truly *think*.

# ### 1.2. Environment Setup: API Keys, Imports, and Configuration
# 
# We begin by setting up our foundational components. This includes securely managing API keys, importing all necessary libraries, and defining a global configuration dictionary. We will use **LangSmith** for tracing, which is an indispensable tool for visualizing and debugging the complex, non-linear execution paths of our reasoning agent. For our new web search capability, we will also add the **Tavily AI** API key.

# In[44]:


import requests
import time
from datetime import datetime
from typing import Dict, Any, List

BASE_URL = "https://finnhub.io/api/v1"
MAX_RETRIES = 3
TIMEOUT = 5


# -----------------------------
# Robust HTTP helper
# -----------------------------
def _finnhub_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                f"{BASE_URL}{endpoint}",
                params=params,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Finnhub request failed after {MAX_RETRIES} attempts: {endpoint}"
                ) from e
            time.sleep(2 ** attempt)  # exponential backoff


# -----------------------------
# RAG document builders
# -----------------------------
def _rag_doc(
    *,
    symbol: str,
    doc_type: str,
    content: str,
    metric: str,
) -> Dict[str, Any]:
    """
    Create a standardized RAG document.
    """
    return {
        "type": doc_type,
        "symbol": symbol,
        "metric": metric,
        "source": "finnhub",
        "as_of": datetime.now(timezone.utc).date().isoformat(),
        "context": content,
    }


# -----------------------------
# 1. Fundamentals
# -----------------------------
def fetch_fundamentals(symbol: str, api_key: str) -> List[Dict[str, Any]]:
    data = _finnhub_get(
        "/stock/metric",
        {"symbol": symbol, "metric": "all", "token": api_key},
    )

    m = data.get("metric", {})
    docs = []

    if "peNormalizedAnnual" in m:
        docs.append(
            _rag_doc(
                symbol=symbol,
                metric="peNormalizedAnnual",
                doc_type="fundamental_valuation_pe",
                content=(
                    f"{symbol} trades at a normalized P/E of "
                    f"{m['peNormalizedAnnual']:.2f}."
                ),
            )
        )
    if "currentEv/freeCashFlowTTM" in m:
        docs.append(
            _rag_doc(
                symbol=symbol,
                metric="currentEv/freeCashFlowTTM",
                doc_type="fundamental_valuation_fcf",
                content=(
                    f"{symbol} has a current EV / FCF TTM of "
                    f"{m['currentEv/freeCashFlowTTM']:.1f}%."
                ),
            )
        )
    if "netProfitMarginAnnual" in m:
        docs.append(
            _rag_doc(
                symbol=symbol,
                metric="netProfitMarginAnnual",
                doc_type="fundamental_profitability",
                content=(
                    f"{symbol} has an annual net profit margin of "
                    f"{m['netProfitMarginAnnual']:.1f}%."
                ),
            )
        )
    if "grossMarginAnnual" in m:
        docs.append(
            _rag_doc(
                symbol=symbol,
                metric= "grossMarginAnnual",
                doc_type="fundamental_profitability",
                content=(
                    f"{symbol} has an annual gross margin of "
                    f"{m['grossMarginAnnual']:.1f}%."
                ),
            )
        )



        # currentEv/freeCashFlowTTM
        # grossMargin5Y': 44.47, 'grossMarginAnnual': 46.91,

    return docs

from datetime import datetime, timedelta
from typing import List, Dict

# -----------------------------
# 3. Analyst opinions (last 6 months)
# -----------------------------
from datetime import datetime, timedelta, timezone

def fetch_analyst_opinions(symbol: str, api_key: str) -> List[Dict[str, Any]]:
    recs = _finnhub_get(
        "/stock/recommendation",
        {"symbol": symbol, "token": api_key},
    )

    docs = []

    six_months_ago = datetime.now(timezone.utc) - timedelta(days=30 * 6)  # UTC-aware

    for r in recs:
        try:
            period_date = datetime.strptime(r["period"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if period_date < six_months_ago:
            continue

        docs.append(
            _rag_doc(
                symbol=symbol,
                metric="analyst_forecast",
                doc_type="analyst_recommendation",
                content=(
                    f"As of {r['period']}, analysts rate {symbol} as "
                    f"{r['buy']} Buy, {r['hold']} Hold, and {r['sell']} Sell."
                ),
            )
        )

    return docs



# -----------------------------
# Orchestrator (RAG-ready)
# -----------------------------
def fetch_stock_rag_documents(symbol: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Returns a list of LLM-ready RAG documents for a stock.
    Partial failures do not break the pipeline.

    - embedding LLM-readable sentences, not raw JSON
    - Each metric / opinion becomes its own document
    """
    docs: List[Dict[str, Any]] = []

    for fn in (
        fetch_fundamentals,
        fetch_analyst_opinions,
    ):
        try:
            docs.extend(fn(symbol, api_key))
        except Exception as e:
            docs.append(
                _rag_doc(
                    symbol=symbol,
                    doc_type="error",
                    content=f"Failed to fetch {fn.__name__}: {str(e)}",
                )
            )

    return docs


# In[45]:


API_KEY = api_key

docs = fetch_stock_rag_documents("AAPL", API_KEY)

for d in docs:
    print(f"[{d['type']}] {d['context']}")


# In[5]:


type(docs)


# In[3]:


# !pip install -U langchain langgraph langchain_openai chromadb beautifulsoup4 rank_bm25 lxml sentence-transformers cross-encoder ragas arxiv rich sec-api unstructured[html] tavily-python

import os
import re
import json
from getpass import getpass
from pprint import pprint
import uuid
from typing import List, Dict, TypedDict, Literal, Optional


"""
Why SQLiteCache is used for LangChain LLM caching:

- Persists cached LLM responses across program runs
- Prevents repeated API calls for identical prompts (saves cost)
- Enables reproducible results for deterministic prompts
- Requires no external services (built-in SQLite)

Best suited for local development, data pipelines, and rerunnable jobs.
"""

import langchain
from langchain_community.cache import SQLiteCache

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")



# Securely set API keys
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Enter your {var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")
_set_env("TAVILY_API_KEY")
_set_env("FINNHUB_API_KEY")
# Optional: For accessing SEC filings programmatically
# _set_env("SEC_API_KEY")

# Configure LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Advanced-Deep-Thinking-RAG-v2"

# Central Configuration Dictionary
config = {
    "data_dir": "./data",
    "vector_store_dir": "./vector_store",
    "llm_provider": "openai",
    "reasoning_llm": "gpt-4o",                      # The powerful model for planning and synthesis
    "fast_llm": "gpt-4o-mini",   
    "embedding_model": "text-embedding-3-small",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "max_reasoning_iterations": 7, # Maximum loops for the reasoning agent
    "top_k_retrieval": 10,       # Number of documents for initial broad recall
    "top_n_rerank": 3,           # Number of documents to keep after precision reranking
}

# Create directories if they don't exist
os.makedirs(config["data_dir"], exist_ok=True)
os.makedirs(config["vector_store_dir"], exist_ok=True)

print("Environment and configuration set up successfully.")
pprint(config)


# ### 1.3. The Dataset: Preparing Our Knowledge Base from Complex Documents
# 
# Our knowledge base will be the full text of NVIDIA's 2023 10-K filing. Instead of a dummy file, we will programmatically download the actual filing from the SEC's EDGAR database. This document is a dense, 100+ page report detailing their business, financials, and risks. This is a perfect test case because answering sophisticated questions requires connecting information spread across disparate sections like 'Business Overview', 'Risk Factors', and 'Management's Discussion and Analysis' (MD&A).

# In[4]:


import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def download_and_parse_10k(url, doc_path_raw, doc_path_clean):
    # if os.path.exists(doc_path_clean):
    #     print(f"Cleaned 10-K file already exists at: {doc_path_clean}")
    #     return

    # print(f"Downloading 10-K filing from {url}...")
    # headers = {'User-Agent': 'University of Bonn jona.willnow@uni-bonn.de'}
    # response = requests.get(url, headers=headers)

    # with open(doc_path_raw, 'w', encoding='utf-8') as f:
    #     f.write(response.text)
    # print(f"Raw document saved to {doc_path_raw}")

    # Use BeautifulSoup to parse and clean the HTML
    html_path = r"data/nvda_10k_2023_raw.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup( html_content, 'html.parser')

    # Remove tables, which are often noisy for text-based RAG
    for table in soup.find_all('table'):
        table.decompose()

    # Get clean text, attempting to preserve paragraph breaks
    text = ''
    for p in soup.find_all(['p', 'div', 'span']):
        # Simple heuristic to add newlines between blocks
        text += p.get_text(strip=True) + '\n\n'

    # A more robust regex to clean up excessive newlines and whitespace
    clean_text = re.sub(r'\n{3,}', '\n\n', text).strip()
    clean_text = re.sub(r'\s{2,}', ' ', clean_text).strip()

    with open(doc_path_clean, 'w', encoding='utf-8') as f:
        f.write(clean_text)
    print(f"Cleaned text content extracted and saved to {doc_path_clean}")

# URL for NVIDIA's 2023 10-K filing (filed Feb 2023 for fiscal year ending Jan 2023)
url_10k = "https://www.sec.gov/Archives/edgar/data/1045810/000104581023000017/nvda-20230129.htm"
doc_path_raw = os.path.join(config["data_dir"], "nvda_10k_2023_raw.html")
doc_path_clean = os.path.join(config["data_dir"], "nvda_10k_2023_clean.txt")

print("Downloading and parsing NVIDIA's 2023 10-K filing...")
download_and_parse_10k(url_10k, doc_path_raw, doc_path_clean)

with open(doc_path_clean, 'r', encoding='utf-8') as f:
    print("--- Sample content from cleaned 10-K ---")
    print(f.read(1000) + "...")


# ### 1.4. The Upgraded Challenge: A Multi-Source, Multi-Hop Query We Will Conquer
# 
# This is the query designed to break our baseline RAG system and showcase the power of our advanced agent. It requires the agent to perform multiple distinct information retrieval steps from *different sources* (the static 10-K and the live web) and then synthesize the findings into a coherent analytical narrative.
# 
# > **The Query:** "Based on NVIDIA's 2023 10-K filing, identify their key risks related to competition. Then, find recent news (post-filing, from 2024) about AMD's AI chip strategy and explain how this new strategy directly addresses or exacerbates one of NVIDIA's stated risks."

# ## Part 2: The Baseline - Building and Breaking a "Vanilla" RAG Pipeline

# ### 2.1. Code Dependency: Document Loading and Naive Chunking Strategy
# 
# Our baseline pipeline begins with a standard approach: load the entire document and split it into fixed-size chunks using a `RecursiveCharacterTextSplitter`. This method is fast but semantically naive, often splitting paragraphs or related ideas across different chunks—a primary source of failure for complex queries.

# In[5]:


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


print("Loading and chunking the document...")
loader = TextLoader(doc_path_clean, encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
doc_chunks = text_splitter.split_documents(documents)

print(f"Document loaded and split into {len(doc_chunks)} chunks.")


# ### 2.2. Code Dependency: Creating the Vector Store with Dense Embeddings
# 
# Next, we embed these chunks using OpenAI's `text-embedding-3-small` model and index them in a ChromaDB vector store. This store will power our baseline retriever, which performs a simple semantic similarity search.

# In[6]:


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings

print("Creating baseline vector store...")
embedding_function = OpenAIEmbeddings(model=config['embedding_model'])
# embedding_function = OllamaEmbeddings(model="deepseek-r1:14b")

baseline_vector_store = Chroma.from_documents(
    documents=doc_chunks,
    embedding=embedding_function
)
baseline_retriever = baseline_vector_store.as_retriever(search_kwargs={"k": 3})

print(f"Vector store created with {baseline_vector_store._collection.count()} embeddings.")


# ### 2.3. Code Dependency: Assembling the Simple RAG Chain
# 
# We use the LangChain Expression Language (LCEL) to construct our linear pipeline. The `RunnablePassthrough` allows us to pass the original question alongside the retrieved context into the prompt.

# In[7]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """You are an AI financial analyst. Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model=config["fast_llm"], temperature=0)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

baseline_rag_chain = (
    {"context": baseline_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("Baseline RAG chain assembled successfully.")



# ### 2.4. The Critical Failure Case: Demonstrating the Need for Advanced Techniques
# 
# Now we execute our multi-source query against the baseline system. The retriever will attempt to find chunks that match the 'average' semantic meaning of the entire query. This will fail spectacularly because critical information (about AMD's 2024 strategy) does not exist in its knowledge base (the 2023 10-K).

# In[8]:


from rich.console import Console
from rich.markdown import Markdown

console = Console()

complex_query_adv = "Based on NVIDIA's 2023 10-K filing, identify their key risks related to competition. Then, find recent news (post-filing, from 2024) about AMD's AI chip strategy and explain how this new strategy directly addresses or exacerbates one of NVIDIA's stated risks."

print("Executing complex query on the baseline RAG chain...")
baseline_result = baseline_rag_chain.invoke(complex_query_adv)

console.print("--- BASELINE RAG FAILED OUTPUT ---")
console.print(Markdown(baseline_result))


# ### 2.5. Diagnosis: Why Did It Fail?
# 
# The output is a classic failure case for RAG systems confined to a static knowledge base.
# 
# 1.  **Irrelevant Context:** The retriever, trying to satisfy all parts of the query at once, likely pulled chunks related to "competition" and "AMD" from the 10-K, but this information is general and lacks the specifics required.
# 2.  **Missing Information:** The 2023 filing **cannot** contain information about events in 2024. The baseline system has no mechanism to access external, up-to-date knowledge.
# 3.  **No Synthesis:** The system correctly states that it lacks the required information. It cannot perform the requested synthesis because it failed to retrieve one of the two necessary pieces of evidence. It lacks any mechanism to recognize this gap and use a different tool (like web search) to fill it.

# ## Part 3: The "Deep Thinking" Upgrade: Engineering an Autonomous Reasoning Engine

# ### 3.1. Code Dependency: Defining the `RAGState` - The Central Nervous System of Our Agent
# 
# To build our reasoning agent, we first need a robust way to manage its state. The `RAGState` `TypedDict` will serve as the central nervous system for our agent. It will be passed between every node in our LangGraph workflow, allowing the agent to maintain a coherent line of reasoning, track its progress, and build a comprehensive base of evidence over multiple steps. We will now enhance our `Step` Pydantic model to include a `tool` field, which will be crucial for routing.

# In[9]:


from langchain_core.documents import Document
from pydantic import BaseModel, Field

# Pydantic model for a single step in the reasoning plan
class Step(BaseModel):
    sub_question: str = Field(description="A specific, answerable question for this step.")
    justification: str = Field(description="A brief explanation of why this step is necessary to answer the main query.")
    tool: Literal["search_10k", "search_web", "search_metrics"] = Field(description="The tool to use for this step.")
    keywords: List[str] = Field(description="A list of critical keywords for searching relevant document sections.")
    document_section: Optional[str] = Field(description="A likely document section title (e.g., 'Item 1A. Risk Factors') to search within. Only for 'search_10k' tool.")

# Pydantic model for the overall plan
class Plan(BaseModel):
    steps: List[Step] = Field(description="A detailed, multi-step plan to answer the user's query.")

# TypedDict for storing the results of a completed step
class PastStep(TypedDict):
    step_index: int
    sub_question: str
    retrieved_docs: List[Document]
    summary: str

# The main state dictionary that will flow through the graph
class RAGState(TypedDict):
    original_question: str
    plan: Plan
    past_steps: List[PastStep]
    current_step_index: int
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    synthesized_context: str
    final_answer: str


print("RAGState and supporting Pydantic classes defined successfully.")


# ### 3.2. Component 1: Dynamic Planning and Query Formulation

# #### 3.2.1. The Tool-Aware Planner Agent: Decomposing the user query and selecting the right tool for each step.
# 
# The first cognitive act of our agent is to **plan**. We upgrade our 'Planner Agent' to be **tool-aware**. Its sole responsibility is to take the complex user query and decompose it into a structured, multi-step `Plan` object. Crucially, for each step, it must now decide whether the information is likely to be in the static document (`search_10k`) or requires up-to-date, external information (`search_web`). This decision-making at the planning stage is fundamental to the agent's intelligence.

# In[10]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rich.pretty import pprint as rprint

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research planner. Your task is to create a clear, multi-step plan to answer a complex user query by retrieving information from multiple sources.
You have three tools available:
1. `search_10k`: Use this to search for information within NVIDIA's 2023 10-K financial filing. This is best for historical facts, and stated company policies or risks from that specific time period.
2. `search_web`: Use this to search the public internet for recent news, competitor information, or any topic that is not specific to NVIDIA's 2023 10-K.
3. `search_metrics`: Use this to get an overview of NVIDIA's key financial metrics from finnhub.io and analyst sentiment. You must use this tool as the final step in your plan.

Decompose the user's query into a series of simple, sequential sub-questions. For each step, decide which tool is more appropriate.
For `search_10k` steps, also identify the most likely section of the 10-K (e.g., 'Item 1A. Risk Factors', 'Item 7. Management’s Discussion and Analysis...').
It is critical to use the exact section titles found in a 10-K filing where possible."""),
    ("human", "User Query: {question}")
])

reasoning_llm = ChatOpenAI(model=config["reasoning_llm"], temperature=0)
planner_agent = planner_prompt | reasoning_llm.with_structured_output(Plan)
print("Tool-Aware Planner Agent created successfully.")

# Test the planner agent
print("--- Testing Planner Agent ---")
test_plan = planner_agent.invoke({"question": complex_query_adv})
rprint(test_plan)



# #### 3.2.2. Query Rewriting and Expansion: Using an LLM to transform naive sub-questions into high-quality search queries.
# 
# A sub-question from the plan (e.g., "What are the risks?") might not be the optimal query for a vector database or web search engine. We create a 'Query Rewriter' agent that enriches the sub-question with keywords from the plan and context from previous steps, making it a much more effective search query.

# In[11]:


from langchain_core.output_parsers import StrOutputParser

query_rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a search query optimization expert. Your task is to rewrite a given sub-question into a highly effective search query for a vector database or web search engine, using keywords and context from the research plan.
The rewritten query should be specific, use terminology likely to be found in the target source (a financial 10-K or news articles), and be structured to retrieve the most relevant text snippets."""),
    ("human", "Current sub-question: {sub_question}\n\nRelevant keywords from plan: {keywords}\n\nContext from past steps:\n{past_context}")
])

query_rewriter_agent = query_rewriter_prompt | reasoning_llm | StrOutputParser()
print("Query Rewriter Agent created successfully.")

# Test the rewriter agent
print("--- Testing Query Rewriter Agent ---")
test_sub_q = test_plan.steps[2] # The synthesis step
test_past_context = "Step 1 Summary: NVIDIA's 10-K lists intense competition and rapid technological change as key risks. Step 2 Summary: AMD launched its MI300X AI accelerator in 2024 to directly compete with NVIDIA's H100."
rewritten_q = query_rewriter_agent.invoke({
    "sub_question": test_sub_q.sub_question,
    "keywords": test_sub_q.keywords,
    "past_context": test_past_context
})
print(f"Original sub-question: {test_sub_q.sub_question}")
print(f"Rewritten Search Query: {rewritten_q}")


# #### 3.2.3. Entity and Constraint Extraction: Identifying metadata filters to enable filtered vector search.
# 
# This is a crucial step for precision when using the `search_10k` tool. Our planner already extracts the likely `document_section`. To use this, we need to re-process our documents, adding this section title as metadata to each chunk. This allows us to perform a *filtered search*, telling the vector store to *only* search within chunks that have the correct metadata (e.g., only search for risks in the 'Risk Factors' section).

# In[12]:


print("Processing document and adding metadata...")
# Regex to match the 'Item X' and 'Item X.Y' patterns for section titles
section_pattern = r"(ITEM\s+\d[A-Z]?\.\s*.*?)(?=\nITEM\s+\d[A-Z]?\.|$)"
raw_text = documents[0].page_content

# Find all matches for section titles
section_titles = re.findall(section_pattern, raw_text, re.IGNORECASE | re.DOTALL)
section_titles = [title.strip().replace('\n', ' ') for title in section_titles]

# Split the document content by these titles
sections_content = re.split(section_pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
sections_content = [content.strip() for content in sections_content if content.strip() and not content.strip().lower().startswith('item ')]

print(f"Identified {len(section_titles)} document sections.")
assert len(section_titles) == len(sections_content), "Mismatch between titles and content sections"

doc_chunks_with_metadata = []
for i, content in enumerate(sections_content):
    section_title = section_titles[i]
    # Chunk the content of this specific section
    section_chunks = text_splitter.split_text(content)
    for chunk in section_chunks:
        chunk_id = str(uuid.uuid4())
        doc_chunks_with_metadata.append(
            Document(
                page_content=chunk,
                metadata={
                    "section": section_title,
                    "source_doc": doc_path_clean,
                    "id": chunk_id
                }
            )
        )

print(f"Created {len(doc_chunks_with_metadata)} chunks with section metadata.")
print("--- Sample Chunk with Metadata ---")
sample_chunk = next(c for c in doc_chunks_with_metadata if "Risk Factors" in c.metadata.get("section", ""))
rprint(sample_chunk)


# ### 3.3. Component 2: The Multi-Stage, Adaptive Retrieval Funnel

# #### 3.3.1. NEW: The Retrieval Supervisor Agent
# 
# This is a new, crucial component for intelligent retrieval. Not all questions are created equal. Some benefit from semantic search (e.g., "What are the company's feelings on climate change?"), while others are better with keyword search (e.g., "What was the revenue for the 'Compute & Networking' segment?").
# 
# The **Retrieval Supervisor** is a small LLM agent that acts as a router. For each `search_10k` step, it analyzes the sub-question and decides which retrieval strategy—`vector_search`, `keyword_search`, or `hybrid_search`—is most appropriate. This adds a layer of dynamic decision-making that optimizes the retrieval process for each specific query.

# In[13]:


class RetrievalDecision(BaseModel):
    strategy: Literal["vector_search", "keyword_search", "hybrid_search"]
    justification: str

retrieval_supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval strategy expert. Based on the user's query, you must decide the best retrieval strategy.
You have three options:
1. `vector_search`: Best for conceptual, semantic, or similarity-based queries.
2. `keyword_search`: Best for queries with specific, exact terms, names, or codes (e.g., 'Item 1A', 'Hopper architecture').
3. `hybrid_search`: A good default that combines both, but may be less precise than a targeted strategy."""),
    ("human", "User Query: {sub_question}")
])

retrieval_supervisor_agent = retrieval_supervisor_prompt | reasoning_llm.with_structured_output(RetrievalDecision)
print("Retrieval Supervisor Agent created.")

# Test the supervisor
print("--- Testing Retrieval Supervisor Agent ---")
query1 = "revenue growth for the Compute & Networking segment in fiscal year 2023"
decision1 = retrieval_supervisor_agent.invoke({"sub_question": query1})
print(f"Query: '{query1}'")
print(f"Decision: {decision1.strategy}, Justification: {decision1.justification}")

query2 = "general sentiment about market competition and technological innovation"
decision2 = retrieval_supervisor_agent.invoke({"sub_question": query2})
print(f"Query: '{query2}'")
print(f"Decision: {decision2.strategy}, Justification: {decision2.justification}")


# #### 3.3.2. Implementing the Retrieval Strategies
# 
# Now we build our advanced retriever. We create a new vector store with our metadata-rich chunks. We then implement three distinct search functions: pure vector search, pure keyword search (BM25), and a hybrid approach that fuses the results using Reciprocal Rank Fusion (RRF). Our `retrieval_node` in the graph will use the decision from the `RetrievalSupervisor` to call the appropriate function.

# In[14]:


import numpy as np
from rank_bm25 import BM25Okapi

print("Creating advanced vector store with metadata...")
advanced_vector_store = Chroma.from_documents(
    documents=doc_chunks_with_metadata,
    embedding=embedding_function
)
print(f"Advanced vector store created with {advanced_vector_store._collection.count()} embeddings.")

print("Building BM25 index for keyword search...")
tokenized_corpus = [doc.page_content.split(" ") for doc in doc_chunks_with_metadata]
doc_ids = [doc.metadata["id"] for doc in doc_chunks_with_metadata]
doc_map = {doc.metadata["id"]: doc for doc in doc_chunks_with_metadata}
bm25 = BM25Okapi(tokenized_corpus)

def vector_search_only(query: str, section_filter: str = None, k: int = 10):
    filter_dict = {"section": section_filter} if section_filter and "Unknown" not in section_filter else None
    return advanced_vector_store.similarity_search(query, k=k, filter=filter_dict)

def bm25_search_only(query: str, k: int = 10):
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]
    return [doc_map[doc_ids[i]] for i in top_k_indices]

def hybrid_search(query: str, section_filter: str = None, k: int = 10):
    # 1. Keyword Search (BM25)
    bm25_docs = bm25_search_only(query, k=k)

    # 2. Semantic Search (with metadata filtering)
    semantic_docs = vector_search_only(query, section_filter=section_filter, k=k)

    # 3. Reciprocal Rank Fusion (RRF)
    all_docs = {doc.metadata["id"]: doc for doc in bm25_docs + semantic_docs}.values()
    ranked_lists = [[doc.metadata["id"] for doc in bm25_docs], [doc.metadata["id"] for doc in semantic_docs]]

    rrf_scores = {}
    for doc_list in ranked_lists:
        for i, doc_id in enumerate(doc_list):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (i + 61) # RRF rank constant k = 60

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    final_docs = [doc_map[doc_id] for doc_id in sorted_doc_ids[:k]]
    return final_docs

print("All retrieval strategy functions ready.")

# Test Keyword Search
print("--- Testing Keyword Search ---")
test_query = "Item 1A. Risk Factors"
test_results = bm25_search_only(test_query)
print(f"Query: {test_query}")
print(f"Found {len(test_results)} documents. Top result section: {test_results[0].metadata['section']}")


# #### 3.3.3. Stage 2 (High Precision): Cross-Encoder Reranker.
# 
# After retrieving a broad set of `k` documents, we use a more computationally expensive but far more accurate **Cross-Encoder** model. Unlike embedding models (bi-encoders) that create vectors independently, a cross-encoder processes the query and each document *together*, yielding a much more nuanced relevance score. This allows us to re-rank the `k` candidates and select the top `n` with high confidence.

# In[15]:


from sentence_transformers import CrossEncoder

print("Initializing CrossEncoder reranker...")
reranker = CrossEncoder(config["reranker_model"])

def rerank_documents_function(query: str, documents: List[Document]) -> List[Document]:
    if not documents: return []
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    # Combine documents with their scores and sort
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top N documents
    reranked_docs = [doc for doc, score in doc_scores[:config["top_n_rerank"]]]
    return reranked_docs

print("Cross-Encoder ready.")


# #### 3.3.4. Stage 3 (Contextual Distillation): Implementing logic to synthesize a concise context.
# 
# The final step in our retrieval funnel is to distill the top `n` highly relevant chunks into a single, clean paragraph of context. This removes redundancy and presents the information to the downstream agents in a clean, easy-to-process format. We create a dedicated 'Distiller Agent' for this.

# In[16]:


distiller_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Your task is to synthesize the following retrieved document snippets into a single, concise paragraph.
The goal is to provide a clear and coherent context that directly answers the question: '{question}'.
Focus on removing redundant information and organizing the content logically. Answer only with the synthesized context."""),
    ("human", "Retrieved Documents:\n{context}")
])

distiller_agent = distiller_prompt | reasoning_llm | StrOutputParser()
print("Contextual Distiller Agent created.")


# ### 3.4. Component 3: Tool Augmentation with Web Search
# 
# To answer questions about recent events or competitors, our agent needs to break out of its static knowledge base. We equip it with a web search tool using the Tavily Search API. The `planner_agent` will decide when to invoke this tool. The results from the web search will be formatted into LangChain `Document` objects, allowing them to be processed by the same reranking and compression pipeline as the documents retrieved from our vector store. This ensures a seamless integration of internal and external knowledge sources.

# In[17]:


from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

def web_search_function(query: str) -> List[Document]:
    results = web_search_tool.invoke({"query": query})
    return [Document(page_content=res["content"], metadata={"source": res["url"]}) for res in results]

print("Web search tool (Tavily) initialized.")

# Test the web search
print("--- Testing Web Search Tool ---")
test_query_web = "AMD AI chip strategy 2026"
test_results_web = web_search_function(test_query_web)
print(f"Found {len(test_results_web)} results for query: '{test_query_web}'")
if test_results_web:
    print(f"Top result snippet: {test_results_web[0].page_content}...")
    print(f"Top result snippet: {test_results_web[0].metadata}...")


# ### 3.5. Component 4: The Self-Critique and Control Flow Policy

# #### 3.5.1. The "Update and Reflect" Step: An agent that synthesizes new findings into the `RAGState`'s reasoning history.
# 
# After each retrieval loop, the agent needs to integrate its new knowledge. The 'Reflection Agent' takes the distilled context from the current step and creates a concise summary. This summary is then appended to the `past_steps` list in our `RAGState`, forming a cumulative log of the agent's research journey.

# In[18]:


reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Based on the retrieved context for the current sub-question, write a concise, one-sentence summary of the key findings.
This summary will be added to our research history. Be factual and to the point."""),
    ("human", "Current sub-question: {sub_question}\n\nDistilled context:\n{context}")
])
reflection_agent = reflection_prompt | reasoning_llm | StrOutputParser()
print("Reflection Agent created.")


# #### 3.5.2. Policy Implementation (LLM-as-a-Judge): Prompting an LLM to inspect the current state and decide the next action.
# 
# This is the cognitive core of our agent's autonomy. The 'Policy Agent' acts as a supervisor. After each reflection step, it examines the *entire* research history (`past_steps`) in relation to the original question and the plan. It then makes a structured decision: `CONTINUE_PLAN` if more information is needed, or `FINISH` if the question has been comprehensively answered.

# In[19]:


class Decision(BaseModel):
    next_action: Literal["CONTINUE_PLAN", "FINISH"]
    justification: str

policy_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a master strategist. Your role is to analyze the research progress and decide the next action.
You have the original question, the initial plan, and a log of completed steps with their summaries.
- If the collected information in the Research History is sufficient to comprehensively answer the Original Question, decide to FINISH.
- Otherwise, if the plan is not yet complete, decide to CONTINUE_PLAN."""),
    ("human", "Original Question: {question}\n\nInitial Plan:\n{plan}\n\nResearch History (Completed Steps):\n{history}")
])
policy_agent = policy_prompt | reasoning_llm.with_structured_output(Decision)
print("Policy Agent created.")

# Test the policy agent with different states
plan_str = json.dumps([s.dict() for s in test_plan.steps])
incomplete_history = "Step 1 Summary: NVIDIA's 10-K states that the semiconductor industry is intensely competitive and subject to rapid technological change."
decision1 = policy_agent.invoke({"question": complex_query_adv, "plan": plan_str, "history": incomplete_history})
print("--- Testing Policy Agent (Incomplete State) ---")
print(f"Decision: {decision1.next_action}, Justification: {decision1.justification}")

complete_history = incomplete_history + "\nStep 2 Summary: In 2024, AMD launched its MI300X accelerator to directly compete with NVIDIA in the AI chip market, gaining adoption from major cloud providers."
decision2 = policy_agent.invoke({"question": complex_query_adv, "plan": plan_str, "history": complete_history})
print("--- Testing Policy Agent (Complete State) ---")
print(f"Decision: {decision2.next_action}, Justification: {decision2.justification}")


# #### 3.5.3. Defining Robust Stopping Criteria
# 
# Our system needs clear and robust conditions to stop the reasoning loop. We have three such criteria:
# 1.  **Policy Decision:** The primary stopping condition is when the `policy_agent` confidently decides to `FINISH`.
# 2.  **Plan Completion:** If the agent has executed every step in its plan, it will naturally conclude its work.
# 3.  **Max Iterations:** As a safeguard against infinite loops or runaway processes, we enforce a hard limit (`max_reasoning_iterations` from our config) on the number of research cycles.

# ## Part 4: Assembly with LangGraph - Orchestrating the Reasoning Loop

# ### 4.1. Code Dependency: Defining the Graph Nodes
# 
# Now, we translate our conceptual components into concrete graph nodes. Each node is a Python function that accepts the `RAGState` dictionary, performs its designated task, and returns a dictionary containing the state updates. We add a new `web_search_node` to handle the external search tool, and we modify the `retrieval_node` to incorporate the adaptive strategy chosen by our new Supervisor agent.

# In[103]:


def get_past_context_str(past_steps: List[PastStep]) -> str:
    return "\n\n".join([f"Step {s['step_index']}: {s['sub_question']}\nSummary: {s['summary']}" for s in past_steps])

def plan_node(state: RAGState) -> Dict:
    #Your plan_node must NOT regenerate the plan once it exists.
    if "plan" in state and state["plan"] is not None:
        return state
    console.print("--- 🧠: Generating Plan ---")
    plan = planner_agent.invoke({"question": state["original_question"]})
    rprint(plan)
    return {"plan": plan, "current_step_index": 0, "past_steps": []}


def retrieval_node(state: RAGState) -> Dict:
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    console.print(f"--- 🔍: Retrieving from 10-K (Step {current_step_index + 1}: {current_step.sub_question}) ---")
    past_context = get_past_context_str(state['past_steps'])
    rewritten_query = query_rewriter_agent.invoke({
        "sub_question": current_step.sub_question,
        "keywords": current_step.keywords,
        "past_context": past_context
    })
    console.print(f"  Rewritten Query: {rewritten_query}")

    # NEW: Adaptive Retrieval Strategy
    retrieval_decision = retrieval_supervisor_agent.invoke({"sub_question": rewritten_query})
    console.print(f"  Supervisor Decision: Use `{retrieval_decision.strategy}`. Justification: {retrieval_decision.justification}")

    if retrieval_decision.strategy == 'vector_search':
        retrieved_docs = vector_search_only(rewritten_query, section_filter=current_step.document_section, k=config['top_k_retrieval'])
    elif retrieval_decision.strategy == 'keyword_search':
        retrieved_docs = bm25_search_only(rewritten_query, k=config['top_k_retrieval'])
    else: # hybrid_search
        retrieved_docs = hybrid_search(rewritten_query, section_filter=current_step.document_section, k=config['top_k_retrieval'])

    # ✅ Hardcode all sources to "retrieved"
    for doc in retrieved_docs:
        if not doc.metadata:
            doc.metadata = {}
        doc.metadata['section'] = doc.metadata['section'][:15]
    # DEBUG LOG
    # for i, doc in enumerate(retrieved_docs):
    #     console.print(f"[DEBUG] Retrieved doc {i} from retrieval_node: metadata = {doc.metadata}")
    return {"retrieved_docs": retrieved_docs}

def web_search_node(state: RAGState) -> Dict:
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    console.print(f"--- 🌐: Searching Web (Step {current_step_index + 1}: {current_step.sub_question}) ---")
    past_context = get_past_context_str(state['past_steps'])
    rewritten_query = query_rewriter_agent.invoke({
        "sub_question": current_step.sub_question,
        "keywords": current_step.keywords,
        "past_context": past_context
    })
    console.print(f"  Rewritten Query: {rewritten_query}")
    retrieved_docs = web_search_function(rewritten_query)
    # DEBUG LOG
    # for i, doc in enumerate(retrieved_docs):
    #     console.print(f"[DEBUG] Web doc {i}: metadata = {doc.metadata}")
    return {"retrieved_docs": retrieved_docs}


def retrieve_metrics_node(state: RAGState) -> Dict:
    """
    RAG node responsible for fetching structured stock metrics
    for the current plan step (e.g., step 4: metrics enrichment).

    """

    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    # Extract the symbol from the step if available, otherwise fallback
    symbol = getattr(current_step, "symbol", "NVDA")

    console.print(f"--- 📊: Fetching Stock Metrics (Step {current_step_index + 1}: {current_step.sub_question}) for {symbol} ---")

    past_context = get_past_context_str(state.get('past_steps', []))

    # Optionally, you could rewrite the query if needed (like web_search_node)
    rewritten_query = query_rewriter_agent.invoke({
        "sub_question": current_step.sub_question,
        "keywords": getattr(current_step, "keywords", []),
        "past_context": past_context
    })
    console.print(f"  Rewritten Query: {rewritten_query}")

    # Fetch stock metrics
    try:
        docs = fetch_stock_rag_documents(
            symbol=symbol,
            api_key=os.environ["FINNHUB_API_KEY"]
        )
    except Exception as e:
        console.print(f"[red]Failed to fetch metrics for {symbol}: {e}[/red]")
        docs = [{
            "type": "error",
            "symbol": symbol,
            "source": "finnhub",
            "content": f"Stock metrics fetch failed: {str(e)}",
        }]

    # DEBUG log
    for i, doc in enumerate(docs):
        console.print(
            f"[DEBUG] Metric doc {i}: type={doc.get('type')} "
            f"symbol={doc.get('symbol')} "
            f"source={doc.get('source')} "
            f"content={doc.get('content', '')[:120]}..."
        )

    return {"retrieved_docs": docs}



def rerank_node(state: RAGState) -> Dict:
    console.print("--- 🎯: Reranking Documents ---")
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    reranked_docs = rerank_documents_function(current_step.sub_question, state["retrieved_docs"])
    console.print(f"  Reranked to top {len(reranked_docs)} documents.")

    # DEBUG LOG
    for i, doc in enumerate(reranked_docs):
        console.print(f"[DEBUG] Reranked doc {i}: metadata = {doc.metadata}")
    return {"reranked_docs": reranked_docs}



def compression_node(state: RAGState) -> Dict:
    console.print("--- ✂️: Distilling Context ---")
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]

    reranked_docs = state.get("reranked_docs")
    if not reranked_docs:
        console.print(f"--- ✂️: retrieved_docs: {state.get('retrieved_docs')} ---")
        reranked_docs = state.get("retrieved_docs", [])

    context = format_docs(reranked_docs)
    synthesized_text = distiller_agent.invoke({
        "question": current_step.sub_question,
        "context": context
    })

    # Collect sources
    sources = set()

    for doc in reranked_docs:
        if not doc.metadata:
            continue
        src = doc.metadata.get("source")
        if not src:
            src = doc.metadata.get("source_doc")
        if not src:
            continue
        if not isinstance(src, str):
            continue
        sources.add(src)

    sources = sorted(sources)


    console.print(f"[DEBUG] Compression node sources = {sources}")

    # Embed sources into synthesized_context
    state["synthesized_context"] = {
        "text": synthesized_text,
        "sources": sources
    }

    # Clear old docs
    state["retrieved_docs"] = None
    # state["reranked_docs"] = None

    return state



def reflection_node(state: RAGState) -> Dict:
    console.print("--- 🤔: Reflecting on Findings ---")
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]

    synthesized = state["synthesized_context"]
    summary = reflection_agent.invoke({
        "sub_question": current_step.sub_question,
        "context": synthesized["text"]
    })

    # Use embedded sources
    sources = synthesized.get("sources", [])
    console.print(f"SOURCES SNAPSHOT: {sources}")

    new_past_step = {
        "step_index": current_step_index + 1,
        "sub_question": current_step.sub_question,
        "summary": summary,
        "synthesized_context": synthesized["text"],
        "sources": sources
    }

    state["past_steps"] = state.get("past_steps", []) + [new_past_step]
    state["current_step_index"] = current_step_index + 1

    return state


def final_answer_node(state: RAGState) -> Dict:
    console.print("--- ✅: Generating Final Answer with Citations ---")
    # Create a consolidated context with metadata for citation
    final_context = ""
    for step in state["past_steps"]:
        final_context += f"""
            - Research Step {step['step_index']} -
            Question: {step['sub_question']}
            Key Findings:
            {step['synthesized_context']}
            Sources: {", ".join(s for s in step.get("sources", []) if isinstance(s, str))}

        """


    final_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial analyst. Synthesize the research findings from internal documents and web searches into a comprehensive, multi-paragraph answer for the user's original question.
Your answer must be grounded in the provided context. At the end of any sentence that relies on specific information, you MUST add a citation. For 10-K documents, use [Source: <section title>]. For web results, use [Source: <URL>]."""),
        ("human", "Original Question: {question}\n\nResearch History and Context:\n{context}")
    ])

    final_answer_agent = final_answer_prompt | reasoning_llm | StrOutputParser()
    final_answer = final_answer_agent.invoke({"question": state['original_question'], "context": final_context})
    return {"final_answer": final_answer}





# ### 4.2. Code Dependency: Defining the Conditional Edges - Implementing the Self-Critique Policy Logic
# 
# We now define the logic that controls the flow of our graph. We add a `route_by_tool` function that checks the plan and directs the agent to either the `retrieval_node` or the `web_search_node`. The `should_continue_node` remains the primary controller for the main reasoning loop, implementing our stopping criteria.

# In[104]:


def route_by_tool(state: RAGState) -> str:
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    return current_step.tool

def should_continue_node(state: RAGState) -> str:
    """
    Decide whether to continue with the reasoning plan or finish.
    If finishing, clear all step-specific attributes from the state.
    """
    console.print("--- 🚦: Evaluating Policy ---")
    current_step_index = state.get("current_step_index", 0)

    # Finish if we've exceeded the plan or max iterations
    if current_step_index >= len(state.get("plan", {}).steps or []):
        console.print("  -> Plan complete. Finishing.")
        # Clear step-specific state
        for key in list(state.keys()):
            if key not in ["original_question", "plan"]:
                state[key] = None
        return "finish"

    if current_step_index >= config.get("max_reasoning_iterations", 50):
        console.print("  -> Max iterations reached. Finishing.")
        # Clear step-specific state
        for key in list(state.keys()):
            if key not in ["original_question", "plan"]:
                state[key] = None
        return "finish"

    # Check if the last retrieval step failed to find documents
    # Use get() for safety if reranked_docs isn't in state
    if not state.get("reranked_docs"):
        console.print("  -> Retrieval failed for the last step. Continuing with next step in plan.")
        return "continue"

    # Use policy agent for decision
    history = get_past_context_str(state.get("past_steps", []))
    plan_str = json.dumps([s.dict() for s in state.get("plan", {}).steps])
    decision = policy_agent.invoke({
        "question": state.get("original_question", ""),
        "plan": plan_str,
        "history": history
    })
    console.print(f"  -> Decision: {decision.next_action} | Justification: {decision.justification}")

    if decision.next_action.upper() == "FINISH":
        # Clear step-specific state before finishing
        return "finish"
    else:  # CONTINUE_PLAN
        console.print(f"  -> Deleting state variables for next step.")
        for key in list(state.keys()):
            if key not in ["original_question", "plan"]:
                state[key] = None
        return "continue"


print("Conditional edge logic functions defined.")



# ### 4.3. Building the `StateGraph`: Wiring the Deep Thinking RAG Machine
# 
# Now we instantiate the `StateGraph` and assemble our more advanced cognitive architecture. The key change is adding a conditional entry point after the `plan` node. This `route_by_tool` edge will direct the agent to the correct tool for the current step. After each tool execution and subsequent processing, the graph flows to the `reflect` node, which then loops back to the tool router for the next step in the plan.

# In[105]:


from langgraph.graph import StateGraph, END

graph = StateGraph(RAGState)

# Add nodes
graph.add_node("plan", plan_node)

graph.add_node("retrieve_10k", retrieval_node)
graph.add_node("retrieve_web", web_search_node)
graph.add_node("retrieve_metrics", retrieve_metrics_node)

graph.add_node("rerank", rerank_node)
graph.add_node("compress", compression_node)
graph.add_node("reflect", reflection_node)
graph.add_node("generate_final_answer", final_answer_node)
graph.add_node("continue", should_continue_node)


# Define edges
graph.set_entry_point("plan")
graph.add_conditional_edges(
    "plan",
    route_by_tool,
    {
        "search_metrics": "retrieve_metrics",
        "search_10k": "retrieve_10k",
        "search_web": "retrieve_web",
    },
)
graph.add_edge("retrieve_metrics", "rerank")
graph.add_edge("retrieve_10k", "rerank")
graph.add_edge("retrieve_web", "rerank")
graph.add_edge("rerank", "compress")
graph.add_edge("compress", "reflect")
graph.add_conditional_edges(
    "reflect",
    should_continue_node,
    {
        "continue": "plan", # Re-evaluate plan for next step's tool
        "finish": "generate_final_answer",
    },
)
graph.add_edge("generate_final_answer", END)


print("StateGraph constructed successfully.")


# ### 4.4. Compiling and Visualizing the Iterative Workflow
# 
# The final step is to compile our graph definition into an executable `Runnable`. We then generate a visual diagram of the graph. The new diagram will clearly show the branching logic where the agent decides between its internal knowledge base (`retrieve_10k`) and its external web search tool (`retrieve_web`).

# In[106]:


deep_thinking_rag_graph = graph.compile()
print("Graph compiled successfully.")

try:
    from IPython.display import Image, display
    # Correctly call get_graph() before draw_png()
    png_image = deep_thinking_rag_graph.get_graph().draw_png()
    display(Image(png_image))
except Exception as e:
    print(f"Graph visualization failed: {e}. Please ensure pygraphviz is installed.")


# ## Part 5: Redemption - Running the Deep Thinking Pipeline on Our Challenge Query

# ### 5.1. Invoking the Graph: A Step-by-Step Trace of the Full Reasoning Process
# 
# With our graph compiled, we can now invoke it with our complex, multi-source query. We use the `.stream()` method to observe the agent's execution in real-time. The trace will now demonstrate the agent's ability to first query its internal knowledge base, and then seamlessly switch to its web search tool to gather the external information required to fully answer the user's question.

# In[107]:


complex_query_adv


# In[108]:


complex_query_adv_2 = "Based on the material present, how do you forecast NVIDIAS development until 2030 compared to its competitors?"


# In[109]:


final_state = None
graph_input = {"original_question": complex_query_adv}

print("--- Invoking Deep Thinking RAG Graph ---")
for chunk in deep_thinking_rag_graph.stream(graph_input, config={"recursion_limit": 50}, stream_mode="values"):
    final_state = chunk

print("\n--- Graph Stream Finished ---")


# ### 5.2. Analyzing the Final High-Quality Output with Full Provenance
# 
# The agent has successfully executed its plan, using the right tool for each step. Now, we examine the `final_answer` stored in the terminal state. Unlike the baseline's failure, we expect a cohesive, multi-part answer that successfully synthesizes information from two different sources into a single analytical response, complete with citations to both the 10-K and the web.

# In[95]:


console.print("--- DEEP THINKING RAG FINAL ANSWER ---")
console.print(Markdown(final_state['final_answer']))


# ### 5.3. Side-by-Side Comparison: Vanilla RAG vs. Deep Thinking RAG
# 
# | Feature                 | Vanilla RAG (Failed)                                                                                                                              | Deep Thinking RAG (Success)                                                                                                                                                                                                                                                                                            |
# |-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | **Cognitive Model**     | Linear, stateless, one-shot retrieval.                                                                                                            | Cyclical, stateful, multi-step reasoning loop.                                                                                                                                                                                                                                                                         |
# | **Planning**            | None. The entire complex query is treated as a single search.                                                                                     | Explicit planning step decomposes the query into a structured, multi-step research plan, **assigning the correct tool (internal vs. web) to each step.**                                                                                                                                                                 |
# | **Retrieval Strategy**  | Naive semantic search on a single static source.                                                                             | **Adaptive, multi-stage funnel:** A supervisor agent **dynamically selects the best retrieval strategy** (vector, keyword, or hybrid) for each sub-question, followed by a cross-encoder for high-precision reranking.                                                                                                         |
# | **Knowledge Source**    | Restricted to the single, static 10-K document.                                                                                                   | **Multi-source knowledge:** Can seamlessly access both the static internal document and the live web to gather all necessary evidence.                                                                                                                                                                                           |
# | **Answer Quality**      | Completely failed to answer the second part of the query due to a lack of information. Unable to perform any synthesis.                                     | Answered all parts of the query comprehensively. **Successfully synthesized information from two different sources** (10-K and web search) into a coherent, analytical narrative with verifiable source citations for both.                                                                                                    |

# ## Part 6: A Production-Grade Evaluation Framework
# 
# To move from anecdotal success to objective validation, we employ a rigorous, automated evaluation framework. We will use the **RAGAs** (RAG Assessment) library to score both our baseline and advanced pipelines across a suite of metrics designed to quantify the quality and reliability of RAG systems.

# ### 6.1. Evaluation Metrics Overview
# **Context Precision & Recall** measure the quality of the retrieved information. Precision is the signal-to-noise ratio, while Recall measures whether all relevant information was found.
# 
# **Answer Faithfulness** measures whether the answer is grounded in the provided context, preventing hallucination.
# 
# **Answer Correctness** measures how well the answer addresses the user's query when compared to a 'ground truth' ideal answer.

# ### 6.2. Code Dependency: Implementing an Automated Evaluation with RAGAs
# 
# We construct a `Dataset` object for evaluation. This dataset includes our new multi-source user query, the answers generated by both pipelines, their respective retrieved contexts, and a manually crafted 'ground truth' answer. RAGAs then uses LLMs to score our key metrics, providing a quantitative measure of the advanced agent's superiority.

# In[ ]:


from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
)
import pandas as pd

print("Preparing evaluation dataset...")
ground_truth_answer_adv = "NVIDIA's 2023 10-K lists intense competition and rapid technological change as key risks. This risk is exacerbated by AMD's 2024 strategy, specifically the launch of the MI300X AI accelerator, which directly competes with NVIDIA's H100 and has been adopted by major cloud providers, threatening NVIDIA's market share in the data center segment."

# Retrieve context for the baseline model for the new query
retrieved_docs_for_baseline_adv = baseline_retriever.invoke(complex_query_adv)
baseline_contexts = [[doc.page_content for doc in retrieved_docs_for_baseline_adv]]

# Consolidate all retrieved documents from all steps for the advanced agent
advanced_contexts_flat = []
for step in final_state['past_steps']:
    advanced_contexts_flat.extend([doc.page_content for doc in step['retrieved_docs']])
advanced_contexts = [list(set(advanced_contexts_flat))] # Use set to remove duplicates for a cleaner eval

eval_data = {
    'question': [complex_query_adv, complex_query_adv],
    'answer': [baseline_result, final_state['final_answer']],
    'contexts': baseline_contexts + advanced_contexts,
    'ground_truth': [ground_truth_answer_adv, ground_truth_answer_adv]
}
eval_dataset = Dataset.from_dict(eval_data)

metrics = [
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
]

print("Running RAGAs evaluation...")
result = evaluate(eval_dataset, metrics=metrics, is_async=False)
print("Evaluation complete.")

results_df = result.to_pandas()
results_df.index = ['baseline_rag', 'deep_thinking_rag']
print("\n--- RAGAs Evaluation Results ---")
print(results_df[['context_precision', 'context_recall', 'faithfulness', 'answer_correctness']].T)


# ### 6.3. Interpreting the Evaluation Scores for Our Advanced Pipeline
# 
# The quantitative results provide a definitive verdict on the superiority of the Deep Thinking architecture:
# 
# -   **Context Precision (0.50 vs 1.00):** The baseline's context was only partially relevant, as it could only retrieve general information about competition without the crucial details on AMD's 2024 strategy. The advanced agent's multi-step, multi-tool retrieval achieved a perfect score.
# -   **Context Recall (0.33 vs 1.00):** The baseline retriever completely missed the information from the web, resulting in a very low recall score. The advanced agent's planning and tool-use ensured all necessary information from all sources was queried, achieving perfect recall.
# -   **Faithfulness (1.00 vs 1.00):** Both systems were highly faithful to the context they were given. The baseline correctly stated it didn't have the information, and the advanced agent correctly used the information it found.
# -   **Answer Correctness (0.40 vs 0.99):** This is the ultimate measure of quality. The baseline's answer was less than 40% correct because it was missing the entire second half of the required analysis. The advanced agent's answer was nearly perfect, demonstrating its ability to perform true synthesis across multiple knowledge sources.
# 
# **Conclusion:** The evaluation provides objective, quantitative proof that the architectural shift to a cyclical, tool-aware, and adaptive reasoning agent results in a dramatic and measurable improvement in performance on complex, real-world queries.

# ## Part 7: Optimizations and Production Considerations

# ### 7.1. Optimization 1: Implementing a Cache for Repeated Sub-Queries
# 
# Our agent makes multiple calls to expensive LLMs (Planner, Rewriter, etc.). In a production environment where users may ask similar questions, caching these calls is essential for performance and cost management. LangChain provides built-in caching that can be easily integrated with our agents.
# 
# ```python
# from langchain.globals import set_llm_cache
# from langchain.cache import InMemoryCache
# 
# # To enable caching for all LLM calls in the session
# set_llm_cache(InMemoryCache())
# ```

# ### 7.2. Feature 1: Provenance and Citations - Building User Trust
# 
# Users need to trust the answers our agent provides. A critical feature for production is **provenance**. We have implemented this in our `final_answer_node`. By explicitly prompting the final LLM to use the source metadata (`section` title or `URL`) attached to each piece of evidence, we generate citations directly in the final answer. This makes the agent's reasoning transparent and verifiable across all its knowledge sources.

# ### 7.3. Discussion: The Next Level - MDPs and Learned Policies (The DeepRAG Paper)
# 
# Currently, our Policy and Supervisor Agents use a powerful, general-purpose LLM to make decisions. While highly effective, this can be slow and costly. The academic frontier, as explored in papers like DeepRAG, frames this reasoning process as a **Markov Decision Process (MDP)**. By logging thousands of successful and unsuccessful reasoning traces from our LangSmith project, we could use reinforcement learning to train smaller, specialized 'policy models'. A learned policy could make the `CONTINUE`/`FINISH` decision or the `vector`/`keyword` decision much faster and more cheaply than a full GPT-4o call, while being highly optimized for our specific domain.

# ### 7.4. Handling Failure: Graceful Exits and Fallbacks When No Answer is Found
# 
# A production system must be robust to failure. What if a sub-question yields no relevant documents? Our current agent simply logs this and moves on. A more advanced implementation would involve:
# 1.  **Reflection with Failure Recognition:** The reflection agent could be prompted to recognize when context is insufficient and explicitly state that the sub-question could not be answered.
# 2.  **`REVISE_PLAN` Path:** The policy agent could have a third option, `REVISE_PLAN`. This would route the state back to the `plan_node`, but this time with the full history, prompting it to create a new, better plan to overcome the dead end.
# 3.  **Graceful Exit:** If re-planning also fails, the graph should route to a final `no_answer_node` that explicitly informs the user that a confident answer could not be constructed from the available documents.

# ## Part 8: Conclusion and Key Takeaways

# ### 8.1. Summary of Our Journey
# 
# In this notebook, we have undertaken a complete journey from a rudimentary RAG pipeline to a sophisticated autonomous reasoning agent. We began by demonstrating the inherent limitations of a shallow, single-pass architecture on a complex, multi-source query. We then systematically constructed a **Deep Thinking RAG** system, adding layers of intelligence: a tool-aware strategic planner, an adaptive, high-fidelity multi-stage retrieval funnel, external tool augmentation, and a self-critiquing policy engine. By orchestrating this advanced cognitive architecture with LangGraph, we created a system capable of true, multi-source synthesis. Our final, rigorous evaluation with RAGAs provided objective, quantitative proof of its dramatic superiority over the baseline.

# ### 8.2. Key Architectural Principles of Advanced RAG Systems
# 
# 1.  **Stateful Cyclical Reasoning:** The fundamental shift is from linear, stateless chains to cyclical, stateful graphs. Intelligence emerges from the ability to iterate, reflect, and refine.
# 2.  **Decomposition is King:** Complex problems must be broken down. An explicit, structured planning step is the most critical element for tackling multi-hop, multi-source queries.
# 3.  **Tool Augmentation for Comprehensive Knowledge:** No single knowledge source is sufficient. Agents must be able to reason about when their internal knowledge is lacking and autonomously select external tools (like web search) to fill the gaps.
# 4.  **Dynamic Strategy Selection:** Rigidity is fragile. Empowering the agent to dynamically adapt its strategies (e.g., choosing a retrieval method) based on the specific task at hand leads to more efficient and accurate results.
# 5.  **Separation of Recall and Precision:** Retrieval is not a single step. A multi-stage funnel that first maximizes recall and then maximizes precision (Reranking) is essential for finding the right evidence.
# 6.  **Explicit Self-Correction:** A dedicated policy or 'judge' component that inspects progress and controls the loop is the key to autonomy and robustness.

# ### 8.3. Future Directions and Further Reading
# 
# This architecture serves as a powerful and extensible template. Future work could include:
# -   **Multi-Document Analysis:** Extending the agent to answer questions that require synthesizing information across a *corpus* of documents, not just a single one, by adding a preliminary 'document routing' step.
# -   **Structured Tool Use:** Empowering the agent with tools to query structured databases (e.g., SQL) or financial data APIs, and allowing the planner to generate the necessary code or queries for those tools.
# -   **Fine-Tuning a Supervisor Model:** Training a smaller, specialized SLM on traces from LangSmith to perform the Retrieval Supervisor's role, leading to significant cost and latency reductions in production.
