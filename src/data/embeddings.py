"""
Embedding utilities for the Deep Thinking RAG pipeline.

This module handles the creation and management of embedding functions
for vector similarity search.
"""

from langchain_openai import OpenAIEmbeddings

from src.config.settings import CONFIG


def get_embedding_function() -> OpenAIEmbeddings:
    """
    Get the configured embedding function for vector operations.

    Returns:
        An OpenAIEmbeddings instance configured with the model from settings.
    """
    return OpenAIEmbeddings(model=CONFIG['embedding_model'])
