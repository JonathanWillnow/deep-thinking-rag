"""
Shallow (baseline) RAG pipeline for the Deep Thinking RAG project.

This module implements a simple, linear RAG chain that serves as
a baseline for comparison with the advanced deep thinking pipeline.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.config.settings import CONFIG
from src.data.processor import format_docs


# Baseline prompt template
BASELINE_TEMPLATE = """You are an AI financial analyst. Answer the question based only on the following context:
{context}

Question: {question}
"""


def create_baseline_rag_chain(retriever):
    """
    Create a simple baseline RAG chain.

    This implements the standard "retrieve -> augment -> generate" pipeline
    that serves as a baseline for demonstrating the limitations of shallow RAG.

    Args:
        retriever: A LangChain retriever to use for document retrieval.

    Returns:
        A LangChain runnable implementing the baseline RAG chain.
    """
    prompt = ChatPromptTemplate.from_template(BASELINE_TEMPLATE)
    llm = ChatOpenAI(model=CONFIG["fast_llm"], temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Baseline RAG chain assembled successfully.")
    return chain


def run_baseline_query(chain, query: str) -> str:
    """
    Run a query through the baseline RAG chain.

    Args:
        chain: The baseline RAG chain runnable.
        query: The user's question.

    Returns:
        The generated answer string.
    """
    return chain.invoke(query)
