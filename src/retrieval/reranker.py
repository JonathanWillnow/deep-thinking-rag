"""
Cross-encoder reranking for the Deep Thinking RAG pipeline.

This module implements high-precision reranking using a cross-encoder
model to refine retrieval results.
"""

from typing import Optional

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from src.config.settings import CONFIG


# Global reranker instance (initialized lazily)
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """
    Get or initialize the global cross-encoder reranker.

    Lazily initializes the reranker on first call.

    Returns:
        The CrossEncoder reranker instance.
    """
    global _reranker

    if _reranker is None:
        print("Initializing CrossEncoder reranker...")
        _reranker = CrossEncoder(CONFIG["reranker_model"])
        print("Cross-Encoder ready.")

    return _reranker


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = None
) -> list[Document]:
    """
    Rerank documents using a cross-encoder for higher precision.

    Cross-encoders process query-document pairs together, providing
    more accurate relevance scores than bi-encoder embeddings.

    Args:
        query: The search query string.
        documents: List of Document objects to rerank.
        top_n: Number of top documents to return. Defaults to CONFIG value.

    Returns:
        List of reranked Document objects, sorted by relevance.
    """
    if not documents:
        return []

    top_n = top_n or CONFIG["top_n_rerank"]
    reranker = get_reranker()

    # Create query-document pairs
    pairs = [(query, doc.page_content) for doc in documents]

    # Get reranker scores
    scores = reranker.predict(pairs)

    # Combine documents with scores and sort
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top N documents
    reranked_docs = [doc for doc, score in doc_scores[:top_n]]

    return reranked_docs
