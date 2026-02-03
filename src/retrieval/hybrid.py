"""
Hybrid search combining vector and BM25 for the Deep Thinking RAG pipeline.

This module implements hybrid search using Reciprocal Rank Fusion (RRF)
to combine semantic and keyword-based retrieval strategies.
"""

from typing import Optional

from langchain_core.documents import Document

from src.config.settings import CONFIG
from src.retrieval.vector_store import vector_search_only, get_doc_map
from src.retrieval.bm25 import bm25_search_only


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60
) -> dict[str, float]:
    """
    Compute Reciprocal Rank Fusion scores for multiple ranked lists.

    RRF combines rankings from multiple retrieval systems by summing
    reciprocal ranks: score = sum(1 / (k + rank)) for each list.

    Args:
        ranked_lists: List of lists containing document IDs in ranked order.
        k: RRF constant, typically 60. Higher values reduce the impact of rank.

    Returns:
        Dictionary mapping document IDs to their RRF scores.
    """
    rrf_scores = {}

    for doc_list in ranked_lists:
        for i, doc_id in enumerate(doc_list):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (i + k + 1)

    return rrf_scores


def hybrid_search(
    query: str,
    section_filter: Optional[str] = None,
    k: int = None
) -> list[Document]:
    """
    Perform hybrid search combining vector and BM25 retrieval.

    Uses Reciprocal Rank Fusion to combine results from semantic
    vector search and keyword-based BM25 search.

    Args:
        query: The search query string.
        section_filter: Optional section title to filter vector search results.
        k: Number of results to return. Defaults to CONFIG value.

    Returns:
        List of Document objects ranked by RRF score.
    """
    k = k or CONFIG['top_k_retrieval']
    doc_map = get_doc_map()

    # 1. Keyword Search (BM25)
    bm25_docs = bm25_search_only(query, k=k)

    # 2. Semantic Search (with optional metadata filtering)
    semantic_docs = vector_search_only(query, section_filter=section_filter, k=k)

    # 3. Combine all unique documents
    all_docs = {doc.metadata["id"]: doc for doc in bm25_docs + semantic_docs}

    # 4. Create ranked lists for RRF
    ranked_lists = [
        [doc.metadata["id"] for doc in bm25_docs],
        [doc.metadata["id"] for doc in semantic_docs]
    ]

    # 5. Compute RRF scores
    rrf_scores = reciprocal_rank_fusion(ranked_lists)

    # 6. Sort by RRF score and return top-k
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    final_docs = [doc_map[doc_id] for doc_id in sorted_doc_ids[:k] if doc_id in doc_map]

    return final_docs
