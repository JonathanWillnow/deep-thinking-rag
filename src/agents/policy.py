"""
Policy and reflection agents for the Deep Thinking RAG pipeline.

This module implements the policy agent (LLM-as-a-Judge) that decides
when to continue or finish the research process, and the reflection
agent that summarizes findings from each step.
"""

from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.config.settings import CONFIG


class Decision(BaseModel):
    """Policy decision on whether to continue or finish research."""

    next_action: Literal["CONTINUE_PLAN", "FINISH"] = Field(
        description="The next action to take: continue with the plan or finish."
    )
    justification: str = Field(
        description="Explanation for the decision."
    )


# Policy agent prompt template
POLICY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a master strategist. Your role is to analyze the research progress and decide the next action.
You have the original question, the initial plan, and a log of completed steps with their summaries.
- If the collected information in the Research History is sufficient to comprehensively answer the Original Question, decide to FINISH.
- Otherwise, if the plan is not yet complete, decide to CONTINUE_PLAN."""),
    ("human", "Original Question: {question}\n\nInitial Plan:\n{plan}\n\nResearch History (Completed Steps):\n{history}")
])


# Reflection agent prompt template
REFLECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Based on the retrieved context for the current sub-question, write a concise, one-sentence summary of the key findings.
This summary will be added to our research history. Be factual and to the point."""),
    ("human", "Current sub-question: {sub_question}\n\nDistilled context:\n{context}")
])


def get_policy_agent():
    """
    Create and return the policy agent.

    The policy agent examines research progress and decides whether
    to continue with the plan or finish the research process.

    Returns:
        A LangChain runnable that produces a Decision object.
    """
    reasoning_llm = ChatOpenAI(model=CONFIG["reasoning_llm"], temperature=0)
    return POLICY_PROMPT | reasoning_llm.with_structured_output(Decision)


def get_reflection_agent():
    """
    Create and return the reflection agent.

    The reflection agent summarizes findings from each research step
    to build a cumulative research history.

    Returns:
        A LangChain runnable that produces summary strings.
    """
    reasoning_llm = ChatOpenAI(model=CONFIG["reasoning_llm"], temperature=0)
    return REFLECTION_PROMPT | reasoning_llm | StrOutputParser()


# Global agent instances
policy_agent = None
reflection_agent = None


def get_or_create_policy_agent():
    """
    Get or create the global policy agent instance.

    Returns:
        The policy agent runnable.
    """
    global policy_agent
    if policy_agent is None:
        policy_agent = get_policy_agent()
    return policy_agent


def get_or_create_reflection_agent():
    """
    Get or create the global reflection agent instance.

    Returns:
        The reflection agent runnable.
    """
    global reflection_agent
    if reflection_agent is None:
        reflection_agent = get_reflection_agent()
    return reflection_agent
