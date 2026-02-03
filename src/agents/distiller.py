"""
Distiller agent for the Deep Thinking RAG pipeline.

This module implements the contextual distiller that synthesizes
retrieved document snippets into concise, coherent context.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.config.settings import CONFIG


# Distiller prompt template
DISTILLER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Your task is to synthesize the following retrieved document snippets into a single, concise paragraph.
The goal is to provide a clear and coherent context that directly answers the question: '{question}'.
Focus on removing redundant information and organizing the content logically. Answer only with the synthesized context."""),
    ("human", "Retrieved Documents:\n{context}")
])


def get_distiller_agent():
    """
    Create and return the distiller agent.

    The distiller synthesizes multiple retrieved document snippets
    into a single, coherent paragraph of context.

    Returns:
        A LangChain runnable that produces synthesized context strings.
    """
    reasoning_llm = ChatOpenAI(model=CONFIG["reasoning_llm"], temperature=0)
    return DISTILLER_PROMPT | reasoning_llm | StrOutputParser()


# Global distiller agent instance
distiller_agent = None


def get_or_create_distiller_agent():
    """
    Get or create the global distiller agent instance.

    Returns:
        The distiller agent runnable.
    """
    global distiller_agent
    if distiller_agent is None:
        distiller_agent = get_distiller_agent()
    return distiller_agent
