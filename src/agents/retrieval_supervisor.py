"""
Retrieval supervisor agent for the Deep Thinking RAG pipeline.

This module implements the supervisor agent that dynamically selects
the optimal retrieval strategy for each sub-question.
"""

from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config.settings import CONFIG


class RetrievalDecision(BaseModel):
    """Decision on which retrieval strategy to use."""

    strategy: Literal["vector_search", "keyword_search", "hybrid_search"] = Field(
        description="The retrieval strategy to use."
    )
    justification: str = Field(
        description="Explanation for why this strategy was chosen."
    )


# Retrieval supervisor prompt template
RETRIEVAL_SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval strategy expert. Based on the user's query, you must decide the best retrieval strategy.
You have three options:
1. `vector_search`: Best for conceptual, semantic, or similarity-based queries.
2. `keyword_search`: Best for queries with specific, exact terms, names, or codes (e.g., 'Item 1A', 'Hopper architecture').
3. `hybrid_search`: A good default that combines both, but may be less precise than a targeted strategy."""),
    ("human", "User Query: {sub_question}")
])


def get_retrieval_supervisor_agent():
    """
    Create and return the retrieval supervisor agent.

    The supervisor analyzes each sub-question and decides the optimal
    retrieval strategy (vector, keyword, or hybrid search).

    Returns:
        A LangChain runnable that produces a RetrievalDecision object.
    """
    reasoning_llm = ChatOpenAI(model=CONFIG["reasoning_llm"], temperature=0)
    return RETRIEVAL_SUPERVISOR_PROMPT | reasoning_llm.with_structured_output(RetrievalDecision)


# Global retrieval supervisor agent instance
retrieval_supervisor_agent = None


def get_or_create_retrieval_supervisor_agent():
    """
    Get or create the global retrieval supervisor agent instance.

    Returns:
        The retrieval supervisor agent runnable.
    """
    global retrieval_supervisor_agent
    if retrieval_supervisor_agent is None:
        retrieval_supervisor_agent = get_retrieval_supervisor_agent()
    return retrieval_supervisor_agent
