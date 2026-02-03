"""
Graph nodes for the Deep Thinking RAG pipeline.

This module implements all the node functions that perform the
cognitive tasks in the LangGraph workflow.
"""

import os
from typing import Dict, List

from rich.console import Console
from rich.pretty import pprint as rprint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.config.settings import CONFIG
from src.graph.state import RAGState, PastStep
from src.data.processor import format_docs
from src.retrieval.vector_store import vector_search_only
from src.retrieval.bm25 import bm25_search_only
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank_documents
from src.agents.planner import get_or_create_planner_agent
from src.agents.retrieval_supervisor import get_or_create_retrieval_supervisor_agent
from src.agents.query_rewriter import get_or_create_query_rewriter_agent
from src.agents.distiller import get_or_create_distiller_agent
from src.agents.policy import get_or_create_policy_agent, get_or_create_reflection_agent
from src.agents.web_search import web_search
from src.agents.metrics_fetcher import fetch_stock_rag_documents, metrics_to_documents


console = Console()


def get_past_context_str(past_steps: List[PastStep]) -> str:
    """
    Format past research steps into a context string.

    Args:
        past_steps: List of completed research steps.

    Returns:
        Formatted string summarizing past research.
    """
    if not past_steps:
        return ""
    return "\n\n".join([
        f"Step {s['step_index']}: {s['sub_question']}\nSummary: {s['summary']}"
        for s in past_steps
    ])


def plan_node(state: RAGState) -> Dict:
    """
    Generate a research plan for the complex query.

    If a plan already exists in the state, returns without modification.
    Otherwise, invokes the planner agent to create a structured plan.

    Args:
        state: The current RAG state.

    Returns:
        State update with the generated plan.
    """
    # Don't regenerate if plan exists
    if state.get("plan") is not None:
        return state

    console.print("--- [bold blue]Planning[/bold blue]: Generating Research Plan ---")

    try:
        planner_agent = get_or_create_planner_agent()
        plan = planner_agent.invoke({"question": state["original_question"]})
        rprint(plan)

        return {
            "plan": plan,
            "current_step_index": 0,
            "past_steps": []
        }
    except Exception as e:
        console.print(f"[red]ERROR in plan_node: {type(e).__name__}: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise


def retrieval_node(state: RAGState) -> Dict:
    """
    Retrieve documents from the 10-K knowledge base.

    Uses the retrieval supervisor to select the optimal search strategy
    (vector, keyword, or hybrid) for the current sub-question.

    Args:
        state: The current RAG state.

    Returns:
        State update with retrieved documents.
    """
    try:
        current_step_index = state["current_step_index"]
        current_step = state["plan"].steps[current_step_index]

        console.print(f"--- [bold green]Retrieving from 10-K[/bold green] (Step {current_step_index + 1}: {current_step.sub_question}) ---")

        # Rewrite query for better retrieval
        past_context = get_past_context_str(state.get('past_steps', []))
        query_rewriter = get_or_create_query_rewriter_agent()
        rewritten_query = query_rewriter.invoke({
            "sub_question": current_step.sub_question,
            "keywords": current_step.keywords,
            "past_context": past_context
        })
        console.print(f"  Rewritten Query: {rewritten_query}")

        # Get retrieval strategy from supervisor
        retrieval_supervisor = get_or_create_retrieval_supervisor_agent()
        retrieval_decision = retrieval_supervisor.invoke({"sub_question": rewritten_query})
        console.print(f"  Supervisor Decision: Use `{retrieval_decision.strategy}`. Justification: {retrieval_decision.justification}")

        # Execute the chosen retrieval strategy
        if retrieval_decision.strategy == 'vector_search':
            retrieved_docs = vector_search_only(
                rewritten_query,
                section_filter=current_step.document_section,
                k=CONFIG['top_k_retrieval']
            )
        elif retrieval_decision.strategy == 'keyword_search':
            retrieved_docs = bm25_search_only(rewritten_query, k=CONFIG['top_k_retrieval'])
        else:  # hybrid_search
            retrieved_docs = hybrid_search(
                rewritten_query,
                section_filter=current_step.document_section,
                k=CONFIG['top_k_retrieval']
            )

        # Truncate section metadata for display
        for doc in retrieved_docs:
            if doc.metadata and 'section' in doc.metadata:
                doc.metadata['section'] = doc.metadata['section'][:15]

        return {"retrieved_docs": retrieved_docs}
    except Exception as e:
        console.print(f"[red]ERROR in retrieval_node: {type(e).__name__}: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise


def web_search_node(state: RAGState) -> Dict:
    """
    Search the web for external information.

    Uses the Tavily search API to retrieve up-to-date information
    from the internet.

    Args:
        state: The current RAG state.

    Returns:
        State update with retrieved web documents.
    """
    try:
        current_step_index = state["current_step_index"]
        current_step = state["plan"].steps[current_step_index]

        console.print(f"--- [bold cyan]Searching Web[/bold cyan] (Step {current_step_index + 1}: {current_step.sub_question}) ---")

        # Rewrite query for web search
        past_context = get_past_context_str(state.get('past_steps', []))
        query_rewriter = get_or_create_query_rewriter_agent()
        rewritten_query = query_rewriter.invoke({
            "sub_question": current_step.sub_question,
            "keywords": current_step.keywords,
            "past_context": past_context
        })
        console.print(f"  Rewritten Query: {rewritten_query}")

        retrieved_docs = web_search(rewritten_query)

        return {"retrieved_docs": retrieved_docs}
    except Exception as e:
        console.print(f"[red]ERROR in web_search_node: {type(e).__name__}: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise


def retrieve_metrics_node(state: RAGState) -> Dict:
    """
    Fetch stock metrics from Finnhub API.

    Retrieves fundamentals and analyst opinions for the specified stock.

    Args:
        state: The current RAG state.

    Returns:
        State update with metric documents.
    """
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]

    # Extract symbol from step or default to NVDA
    symbol = getattr(current_step, "symbol", "NVDA")

    console.print(f"--- [bold magenta]Fetching Stock Metrics[/bold magenta] (Step {current_step_index + 1}: {current_step.sub_question}) for {symbol} ---")

    past_context = get_past_context_str(state.get('past_steps', []))
    query_rewriter = get_or_create_query_rewriter_agent()
    rewritten_query = query_rewriter.invoke({
        "sub_question": current_step.sub_question,
        "keywords": getattr(current_step, "keywords", []),
        "past_context": past_context
    })
    console.print(f"  Rewritten Query: {rewritten_query}")

    try:
        metrics = fetch_stock_rag_documents(
            symbol=symbol,
            api_key=os.environ.get("FINNHUB_API_KEY")
        )
        docs = metrics_to_documents(metrics)
    except Exception as e:
        console.print(f"[red]Failed to fetch metrics for {symbol}: {e}[/red]")
        docs = []

    return {"retrieved_docs": docs}


def rerank_node(state: RAGState) -> Dict:
    """
    Rerank retrieved documents using a cross-encoder.

    Applies high-precision reranking to select the most relevant
    documents from the initial retrieval.

    Args:
        state: The current RAG state.

    Returns:
        State update with reranked documents.
    """
    console.print("--- [bold yellow]Reranking Documents[/bold yellow] ---")

    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]

    retrieved_docs = state.get("retrieved_docs", [])

    # Handle case where retrieved_docs are dictionaries (from metrics)
    if retrieved_docs and isinstance(retrieved_docs[0], dict):
        from langchain_core.documents import Document
        retrieved_docs = [
            Document(page_content=d.get("context", d.get("content", "")), metadata=d)
            for d in retrieved_docs
        ]

    reranked_docs = rerank_documents(current_step.sub_question, retrieved_docs)
    console.print(f"  Reranked to top {len(reranked_docs)} documents.")

    return {"reranked_docs": reranked_docs}


def compression_node(state: RAGState) -> Dict:
    """
    Distill retrieved documents into concise context.

    Synthesizes multiple document snippets into a coherent paragraph
    and tracks source information.

    Args:
        state: The current RAG state.

    Returns:
        State update with synthesized context.
    """
    console.print("--- [bold white]Distilling Context[/bold white] ---")

    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]

    reranked_docs = state.get("reranked_docs")
    if not reranked_docs:
        reranked_docs = state.get("retrieved_docs", [])

    context = format_docs(reranked_docs)

    distiller = get_or_create_distiller_agent()
    synthesized_text = distiller.invoke({
        "question": current_step.sub_question,
        "context": context
    })

    # Collect sources
    sources = set()
    for doc in reranked_docs:
        if not hasattr(doc, 'metadata') or not doc.metadata:
            continue
        src = doc.metadata.get("source") or doc.metadata.get("source_doc")
        if src and isinstance(src, str):
            sources.add(src)

    sources = sorted(sources)

    return {
        "synthesized_context": {
            "text": synthesized_text,
            "sources": sources
        },
        "retrieved_docs": None
    }


def reflection_node(state: RAGState) -> Dict:
    """
    Reflect on findings and update research history.

    Creates a summary of the current step's findings and appends it
    to the cumulative research history.

    Args:
        state: The current RAG state.

    Returns:
        State update with new past step and incremented index.
    """
    console.print("--- [bold blue]Reflecting on Findings[/bold blue] ---")

    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]

    synthesized = state["synthesized_context"]

    reflection_agent = get_or_create_reflection_agent()
    summary = reflection_agent.invoke({
        "sub_question": current_step.sub_question,
        "context": synthesized["text"]
    })

    sources = synthesized.get("sources", [])
    console.print(f"  Sources: {sources}")

    new_past_step: PastStep = {
        "step_index": current_step_index + 1,
        "sub_question": current_step.sub_question,
        "summary": summary,
        "synthesized_context": synthesized["text"],
        "sources": sources
    }

    past_steps = state.get("past_steps", []) + [new_past_step]

    return {
        "past_steps": past_steps,
        "current_step_index": current_step_index + 1
    }


def final_answer_node(state: RAGState) -> Dict:
    """
    Generate the final comprehensive answer with citations.

    Synthesizes all research findings into a multi-paragraph answer
    with source citations.

    Args:
        state: The current RAG state.

    Returns:
        State update with the final answer.
    """
    try:
        console.print("--- [bold green]Generating Final Answer with Citations[/bold green] ---")

        # Build consolidated context with citations
        final_context = ""
        for step in state["past_steps"]:
            sources_str = ", ".join(s for s in step.get("sources", []) if isinstance(s, str))
            final_context += f"""
- Research Step {step['step_index']} -
Question: {step['sub_question']}
Key Findings:
{step['synthesized_context']}
Sources: {sources_str}

"""

        final_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial analyst. Synthesize the research findings from internal documents and web searches into a comprehensive, multi-paragraph answer for the user's original question.
Your answer must be grounded in the provided context. At the end of any sentence that relies on specific information, you MUST add a citation. For 10-K documents, use [Source: <section title>]. For web results, use [Source: <URL>]."""),
            ("human", "Original Question: {question}\n\nResearch History and Context:\n{context}")
        ])

        reasoning_llm = ChatOpenAI(model=CONFIG["reasoning_llm"], temperature=0)
        final_answer_agent = final_answer_prompt | reasoning_llm | StrOutputParser()

        final_answer = final_answer_agent.invoke({
            "question": state['original_question'],
            "context": final_context
        })

        return {"final_answer": final_answer}
    except Exception as e:
        console.print(f"[red]ERROR in final_answer_node: {type(e).__name__}: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise
