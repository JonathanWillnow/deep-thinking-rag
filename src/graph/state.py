"""
State definitions for the Deep Thinking RAG pipeline.

This module defines the TypedDict classes that represent the state
flowing through the LangGraph workflow.
"""

from typing import List, TypedDict, Optional

from langchain_core.documents import Document

from src.agents.planner import Plan


class PastStep(TypedDict):
    """Record of a completed research step."""

    step_index: int
    sub_question: str
    summary: str
    synthesized_context: str
    sources: List[str]


class RAGState(TypedDict, total=False):
    """
    The main state dictionary that flows through the graph.

    This represents the agent's complete cognitive state including
    the original query, research plan, history, and intermediate results.
    """

    # Core query
    original_question: str

    # Planning
    plan: Optional[Plan]

    # Step tracking
    current_step_index: int
    past_steps: List[PastStep]

    # Retrieval results
    retrieved_docs: Optional[List[Document]]
    reranked_docs: Optional[List[Document]]

    # Synthesized content
    synthesized_context: Optional[dict]  # {"text": str, "sources": List[str]}

    # Final output
    final_answer: Optional[str]
