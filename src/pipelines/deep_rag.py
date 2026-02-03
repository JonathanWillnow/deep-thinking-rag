"""
Deep Thinking RAG pipeline for advanced multi-hop reasoning.

This module provides the main interface for running the Deep Thinking
RAG pipeline, which uses agentic reasoning to handle complex queries.
"""

from typing import Optional, Generator, Any

from rich.console import Console
from rich.markdown import Markdown

from src.graph.workflow import compile_graph
from src.graph.state import RAGState


console = Console()

# Global compiled graph instance
_compiled_graph = None


def get_compiled_graph():
    """
    Get or compile the Deep Thinking RAG graph.

    Lazily compiles the graph on first call for efficiency.

    Returns:
        The compiled LangGraph runnable.
    """
    global _compiled_graph

    if _compiled_graph is None:
        _compiled_graph = compile_graph()

    return _compiled_graph


def run_deep_thinking_rag(
    query: str,
    stream: bool = True,
    recursion_limit: int = 50
) -> RAGState:
    """
    Run a query through the Deep Thinking RAG pipeline.

    Args:
        query: The complex user query to answer.
        stream: Whether to stream intermediate states (default True).
        recursion_limit: Maximum recursion depth for the graph.

    Returns:
        The final RAGState containing the answer and all intermediate results.
    """
    graph = get_compiled_graph()
    graph_input = {"original_question": query}

    final_state = None

    console.print("--- [bold]Invoking Deep Thinking RAG Graph[/bold] ---")

    if stream:
        for chunk in graph.stream(
            graph_input,
            config={"recursion_limit": recursion_limit},
            stream_mode="values"
        ):
            final_state = chunk
    else:
        final_state = graph.invoke(
            graph_input,
            config={"recursion_limit": recursion_limit}
        )

    console.print("\n--- [bold]Graph Execution Finished[/bold] ---")

    return final_state


def stream_deep_thinking_rag(
    query: str,
    recursion_limit: int = 50
) -> Generator[RAGState, None, None]:
    """
    Stream the Deep Thinking RAG pipeline execution.

    Yields intermediate states as the graph executes, allowing
    for real-time monitoring of the reasoning process.

    Args:
        query: The complex user query to answer.
        recursion_limit: Maximum recursion depth for the graph.

    Yields:
        RAGState dictionaries at each step of execution.
    """
    graph = get_compiled_graph()
    graph_input = {"original_question": query}

    for chunk in graph.stream(
        graph_input,
        config={"recursion_limit": recursion_limit},
        stream_mode="values"
    ):
        yield chunk


def display_final_answer(state: RAGState) -> None:
    """
    Display the final answer from a completed RAG state.

    Args:
        state: The final RAGState containing the answer.
    """
    console.print("\n--- [bold green]DEEP THINKING RAG FINAL ANSWER[/bold green] ---\n")

    if state and "final_answer" in state:
        console.print(Markdown(state["final_answer"]))
    else:
        console.print("[red]No final answer available.[/red]")


def get_answer(state: RAGState) -> Optional[str]:
    """
    Extract the final answer from a RAG state.

    Args:
        state: The RAGState to extract the answer from.

    Returns:
        The final answer string, or None if not available.
    """
    if state:
        return state.get("final_answer")
    return None
