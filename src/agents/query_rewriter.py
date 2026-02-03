"""
Query rewriter agent for the Deep Thinking RAG pipeline.

This module implements the query rewriter that transforms sub-questions
into optimized search queries for better retrieval performance.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.config.settings import CONFIG


# Query rewriter prompt template
QUERY_REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a search query optimization expert. Your task is to rewrite a given sub-question into a highly effective search query for a vector database or web search engine, using keywords and context from the research plan.
The rewritten query should be specific, use terminology likely to be found in the target source (a financial 10-K or news articles), and be structured to retrieve the most relevant text snippets."""),
    ("human", "Current sub-question: {sub_question}\n\nRelevant keywords from plan: {keywords}\n\nContext from past steps:\n{past_context}")
])


def get_query_rewriter_agent():
    """
    Create and return the query rewriter agent.

    The query rewriter transforms naive sub-questions into optimized
    search queries using keywords and context from the research plan.

    Returns:
        A LangChain runnable that produces optimized query strings.
    """
    reasoning_llm = ChatOpenAI(model=CONFIG["reasoning_llm"], temperature=0)
    return QUERY_REWRITER_PROMPT | reasoning_llm | StrOutputParser()


# Global query rewriter agent instance
query_rewriter_agent = None


def get_or_create_query_rewriter_agent():
    """
    Get or create the global query rewriter agent instance.

    Returns:
        The query rewriter agent runnable.
    """
    global query_rewriter_agent
    if query_rewriter_agent is None:
        query_rewriter_agent = get_query_rewriter_agent()
    return query_rewriter_agent
