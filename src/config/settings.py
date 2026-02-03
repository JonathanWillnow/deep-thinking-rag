"""
Configuration settings for the Deep Thinking RAG pipeline.

This module centralizes all configuration constants including model names,
chunk sizes, API settings, and directory paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Central Configuration Dictionary
CONFIG = {
    # Directory paths
    "data_dir": str(PROJECT_ROOT / "data"),
    "raw_data_dir": str(PROJECT_ROOT / "data" / "raw"),
    "processed_data_dir": str(PROJECT_ROOT / "data" / "processed"),
    "vector_store_dir": str(PROJECT_ROOT / "vector_store"),

    # LLM settings
    "llm_provider": "openai",
    "reasoning_llm": "gpt-4o",           # Powerful model for planning and synthesis
    "fast_llm": "gpt-4o-mini",           # Fast model for simple tasks

    # Embedding settings
    "embedding_model": "text-embedding-3-small",

    # Reranker settings
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",

    # Retrieval settings
    "top_k_retrieval": 10,               # Documents for initial broad recall
    "top_n_rerank": 3,                   # Documents to keep after reranking

    # Agent settings
    "max_reasoning_iterations": 7,       # Maximum loops for reasoning agent

    # Chunking settings
    "chunk_size": 1000,
    "chunk_overlap": 150,
}


def setup_environment():
    """
    Set up environment variables and create necessary directories.

    Configures LangSmith tracing and ensures all required directories exist.
    """
    # Configure LangSmith tracing
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "Advanced-Deep-Thinking-RAG-v2"

    # Create directories if they don't exist
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    os.makedirs(CONFIG["raw_data_dir"], exist_ok=True)
    os.makedirs(CONFIG["processed_data_dir"], exist_ok=True)
    os.makedirs(CONFIG["vector_store_dir"], exist_ok=True)


def get_api_key(key_name: str) -> str:
    """
    Retrieve an API key from environment variables.

    Args:
        key_name: The name of the environment variable containing the API key.

    Returns:
        The API key value.

    Raises:
        ValueError: If the API key is not set.
    """
    value = os.environ.get(key_name)
    if not value:
        raise ValueError(f"Environment variable {key_name} is not set. Please set it in your .env file.")
    return value
