"""
BM25 keyword search for the Deep Thinking RAG pipeline.

This module implements BM25Okapi-based keyword search as an alternative
to semantic vector search.
"""

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.config.settings import CONFIG


# Global BM25 index (initialized lazily)
_bm25_index: Optional[BM25Okapi] = None
_doc_ids: list = []
_doc_map: dict = {}


def initialize_bm25_index(documents: list[Document]) -> BM25Okapi:
    """
    Initialize the global BM25 index from documents.

    Creates a tokenized corpus and builds the BM25Okapi index for
    keyword-based retrieval.

    Args:
        documents: List of Document objects with 'id' in metadata.

    Returns:
        The initialized BM25Okapi index.
    """
    global _bm25_index, _doc_ids, _doc_map

    print("Building BM25 index for keyword search...")

    # Tokenize corpus
    tokenized_corpus = [doc.page_content.split(" ") for doc in documents]

    # Store document IDs and map
    _doc_ids = [doc.metadata["id"] for doc in documents]
    _doc_map = {doc.metadata["id"]: doc for doc in documents}

    # Build BM25 index
    _bm25_index = BM25Okapi(tokenized_corpus)

    print(f"BM25 index built with {len(documents)} documents.")
    return _bm25_index


def get_bm25_index() -> BM25Okapi:
    """
    Get the global BM25 index instance.

    Returns:
        The BM25Okapi index.

    Raises:
        RuntimeError: If the index has not been initialized.
    """
    if _bm25_index is None:
        raise RuntimeError("BM25 index not initialized. Call initialize_bm25_index first.")
    return _bm25_index


def bm25_search_only(query: str, k: int = None) -> list[Document]:
    """
    Perform BM25 keyword search.

    Args:
        query: The search query string.
        k: Number of results to return. Defaults to CONFIG value.

    Returns:
        List of matching Document objects ranked by BM25 score.
    """
    k = k or CONFIG['top_k_retrieval']
    bm25 = get_bm25_index()

    # Tokenize query
    tokenized_query = query.split(" ")

    # Get BM25 scores
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]

    # Return documents
    return [_doc_map[_doc_ids[i]] for i in top_k_indices]
