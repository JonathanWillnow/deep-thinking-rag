"""
Planner agent for the Deep Thinking RAG pipeline.

This module implements the tool-aware planner that decomposes complex
queries into structured, multi-step research plans.
"""

from typing import List, Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config.settings import CONFIG


class Step(BaseModel):
    """A single step in the research plan."""

    sub_question: str = Field(
        description="A specific, answerable question for this step."
    )
    justification: str = Field(
        description="A brief explanation of why this step is necessary to answer the main query."
    )
    tool: Literal["search_10k", "search_web", "search_metrics"] = Field(
        description="The tool to use for this step."
    )
    keywords: List[str] = Field(
        description="A list of critical keywords for searching relevant document sections."
    )
    document_section: Optional[str] = Field(
        default=None,
        description="A likely document section title (e.g., 'Item 1A. Risk Factors') to search within. Only for 'search_10k' tool."
    )


class Plan(BaseModel):
    """A multi-step research plan to answer a complex query."""

    steps: List[Step] = Field(
        description="A detailed, multi-step plan to answer the user's query."
    )


# Planner prompt template
PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research planner. Your task is to create a clear, multi-step plan to answer a complex user query by retrieving information from multiple sources.
You have three tools available:
1. `search_10k`: Use this to search for information within NVIDIA's 2023 10-K financial filing. This is best for historical facts, and stated company policies or risks from that specific time period.
2. `search_web`: Use this to search the public internet for recent news, competitor information, or any topic that is not specific to NVIDIA's 2023 10-K.
3. `search_metrics`: Use this to get an overview of NVIDIA's key financial metrics from finnhub.io and analyst sentiment. You must use this tool as the final step in your plan.

Decompose the user's query into a series of simple, sequential sub-questions. For each step, decide which tool is more appropriate.
For `search_10k` steps, also identify the most likely section of the 10-K (e.g., 'Item 1A. Risk Factors', 'Item 7. Management's Discussion and Analysis...').
It is critical to use the exact section titles found in a 10-K filing where possible."""),
    ("human", "User Query: {question}")
])


def get_planner_agent():
    """
    Create and return the planner agent.

    The planner agent decomposes complex queries into structured research plans,
    selecting the appropriate tool for each step.

    Returns:
        A LangChain runnable that produces a Plan object.
    """
    reasoning_llm = ChatOpenAI(model=CONFIG["reasoning_llm"], temperature=0)
    return PLANNER_PROMPT | reasoning_llm.with_structured_output(Plan)


# Global planner agent instance
planner_agent = None


def get_or_create_planner_agent():
    """
    Get or create the global planner agent instance.

    Returns:
        The planner agent runnable.
    """
    global planner_agent
    if planner_agent is None:
        planner_agent = get_planner_agent()
    return planner_agent
