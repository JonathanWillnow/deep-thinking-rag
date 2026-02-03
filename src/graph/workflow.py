"""
LangGraph workflow for the Deep Thinking RAG pipeline.

This module defines the conditional edges and constructs the
StateGraph that orchestrates the reasoning loop.
"""

import json
from typing import Literal

from rich.console import Console
from langgraph.graph import StateGraph, END

from src.config.settings import CONFIG
from src.graph.state import RAGState
from src.graph.nodes import (
    plan_node,
    retrieval_node,
    web_search_node,
    retrieve_metrics_node,
    rerank_node,
    compression_node,
    reflection_node,
    final_answer_node,
    get_past_context_str,
)
from src.agents.policy import get_or_create_policy_agent


console = Console()


def route_by_tool(state: RAGState) -> Literal["search_10k", "search_web", "search_metrics"]:
    """
    Route to the appropriate retrieval tool based on the current plan step.

    Args:
        state: The current RAG state.

    Returns:
        The tool identifier for the current step.
    """
    current_step_index = state["current_step_index"]
    current_step = state["plan"].steps[current_step_index]
    return current_step.tool


def should_continue(state: RAGState) -> Literal["continue", "finish"]:
    """
    Decide whether to continue with the plan or finish.

    Implements the stopping criteria:
    1. Plan completion - all steps executed
    2. Max iterations reached
    3. Policy agent decides to finish

    Args:
        state: The current RAG state.

    Returns:
        "continue" to proceed with next step, "finish" to generate final answer.
    """
    console.print("--- [bold red]Evaluating Policy[/bold red] ---")

    current_step_index = state.get("current_step_index", 0)
    plan = state.get("plan")

    # Check plan completion
    if plan and current_step_index >= len(plan.steps):
        console.print("  -> Plan complete. Finishing.")
        return "finish"

    # Check max iterations
    if current_step_index >= CONFIG.get("max_reasoning_iterations", 50):
        console.print("  -> Max iterations reached. Finishing.")
        return "finish"

    # Check if retrieval failed
    if not state.get("reranked_docs"):
        console.print("  -> Retrieval returned no results. Continuing with next step.")
        return "continue"

    # Use policy agent for decision
    history = get_past_context_str(state.get("past_steps", []))
    plan_str = json.dumps([s.model_dump() for s in plan.steps]) if plan else "[]"

    policy_agent = get_or_create_policy_agent()
    decision = policy_agent.invoke({
        "question": state.get("original_question", ""),
        "plan": plan_str,
        "history": history
    })

    console.print(f"  -> Decision: {decision.next_action} | Justification: {decision.justification}")

    if decision.next_action.upper() == "FINISH":
        return "finish"
    else:
        return "continue"


def build_deep_thinking_rag_graph() -> StateGraph:
    """
    Build and return the Deep Thinking RAG StateGraph.

    Constructs the graph with all nodes and edges that implement
    the cyclical reasoning workflow.

    Returns:
        An uncompiled StateGraph ready for compilation.
    """
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

    # Set entry point
    graph.set_entry_point("plan")

    # Add conditional edge from plan to appropriate retrieval tool
    graph.add_conditional_edges(
        "plan",
        route_by_tool,
        {
            "search_10k": "retrieve_10k",
            "search_web": "retrieve_web",
            "search_metrics": "retrieve_metrics",
        },
    )

    # Add edges from retrieval to rerank
    graph.add_edge("retrieve_10k", "rerank")
    graph.add_edge("retrieve_web", "rerank")
    graph.add_edge("retrieve_metrics", "rerank")

    # Add edges through the processing pipeline
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "reflect")

    # Add conditional edge from reflect to either continue or finish
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "continue": "plan",
            "finish": "generate_final_answer",
        },
    )

    # Add final edge to END
    graph.add_edge("generate_final_answer", END)

    print("StateGraph constructed successfully.")
    return graph


def compile_graph():
    """
    Build and compile the Deep Thinking RAG graph.

    Returns:
        A compiled LangGraph runnable ready for execution.
    """
    graph = build_deep_thinking_rag_graph()
    compiled_graph = graph.compile()
    print("Graph compiled successfully.")
    return compiled_graph
