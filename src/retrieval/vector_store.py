"""
Vector store operations for the Deep Thinking RAG pipeline.

This module handles FAISS vector store creation, indexing, and
semantic similarity search operations.
"""

from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config.settings import CONFIG
from src.data.embeddings import get_embedding_function


# Global reference to the advanced vector store (initialized lazily)
_advanced_vector_store: Optional[FAISS] = None
_doc_map: dict = {}


def create_vector_store(documents: list[Document]) -> FAISS:
    """
    Create a FAISS vector store from a list of documents.

    Args:
        documents: List of Document objects to index.

    Returns:
        A FAISS vector store instance.
    """
    try:
        print("Creating vector store...", flush=True)
        embedding_function = get_embedding_function()
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embedding_function
        )
        print(f"Vector store created with {len(documents)} embeddings.", flush=True)
        return vector_store
    except Exception as e:
        print(f"ERROR creating vector store: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def initialize_advanced_vector_store(documents: list[Document]) -> FAISS:
    """
    Initialize the global advanced vector store with metadata-rich documents.

    This creates a vector store optimized for filtered retrieval with
    section metadata.

    Args:
        documents: List of Document objects with section metadata.

    Returns:
        The initialized FAISS vector store.
    """
    global _advanced_vector_store, _doc_map

    print(f"Creating embeddings for {len(documents)} documents...", flush=True)

    try:
        embedding_function = get_embedding_function()

        # Test embedding first
        print("Testing embedding with first document...", flush=True)
        test_text = documents[0].page_content[:500]
        test_embedding = embedding_function.embed_query(test_text)
        print(f"Test embedding successful: {len(test_embedding)} dimensions", flush=True)

        print("Creating FAISS vector store (this may take a moment)...", flush=True)
        _advanced_vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embedding_function
        )
        print("FAISS vector store created successfully!", flush=True)

        # Build document map for quick lookup
        _doc_map = {doc.metadata["id"]: doc for doc in documents}

        print(f"Advanced vector store ready with {len(documents)} embeddings.", flush=True)
        return _advanced_vector_store

    except Exception as e:
        print(f"ERROR in vector store creation: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def get_advanced_vector_store() -> FAISS:
    """
    Get the global advanced vector store instance.

    Returns:
        The advanced FAISS vector store.

    Raises:
        RuntimeError: If the vector store has not been initialized.
    """
    if _advanced_vector_store is None:
        raise RuntimeError("Advanced vector store not initialized. Call initialize_advanced_vector_store first.")
    return _advanced_vector_store


def get_doc_map() -> dict:
    """
    Get the document map for quick ID-based lookups.

    Returns:
        Dictionary mapping document IDs to Document objects.
    """
    return _doc_map


def vector_search_only(
    query: str,
    section_filter: Optional[str] = None,
    k: int = None
) -> list[Document]:
    """
    Perform vector similarity search with optional section filtering.

    Args:
        query: The search query string.
        section_filter: Optional section title to filter results.
        k: Number of results to return. Defaults to CONFIG value.

    Returns:
        List of matching Document objects.
    """
    try:
        k = k or CONFIG['top_k_retrieval']
        vector_store = get_advanced_vector_store()

        # FAISS doesn't support native filtering, so we get more results and filter
        results = vector_store.similarity_search(query, k=k * 2)

        # Apply section filter if specified
        if section_filter and "Unknown" not in section_filter:
            results = [doc for doc in results if doc.metadata.get("section") == section_filter]

        return results[:k]
    except Exception as e:
        print(f"ERROR in vector search: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
