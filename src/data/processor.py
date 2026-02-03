"""
Document processing utilities for the Deep Thinking RAG pipeline.

This module handles text splitting, chunking, and metadata extraction
for documents in the RAG knowledge base.
"""

import re
import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CONFIG


def create_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with the specified or default configuration.

    Args:
        chunk_size: Size of each text chunk. Defaults to CONFIG value.
        chunk_overlap: Overlap between chunks. Defaults to CONFIG value.

    Returns:
        A configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CONFIG["chunk_size"],
        chunk_overlap=chunk_overlap or CONFIG["chunk_overlap"]
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks using the default text splitter.

    Args:
        documents: List of documents to split.

    Returns:
        List of document chunks.
    """
    text_splitter = create_text_splitter()
    return text_splitter.split_documents(documents)


def extract_sections_with_metadata(
    documents: list[Document],
    source_path: str
) -> list[Document]:
    """
    Extract sections from a 10-K document and add metadata to each chunk.

    This function identifies section titles (e.g., 'Item 1A. Risk Factors')
    and creates chunks with section metadata for filtered retrieval.

    Args:
        documents: List of loaded documents (typically one document).
        source_path: Path to the source document for metadata.

    Returns:
        List of Document chunks with section metadata.
    """
    text_splitter = create_text_splitter()

    # Regex to match 'Item X' and 'Item X.Y' patterns for section titles
    section_pattern = r"(ITEM\s+\d[A-Z]?\.\s*.*?)(?=\nITEM\s+\d[A-Z]?\.|$)"
    raw_text = documents[0].page_content

    # Find all matches for section titles
    section_titles = re.findall(section_pattern, raw_text, re.IGNORECASE | re.DOTALL)
    section_titles = [title.strip().replace('\n', ' ') for title in section_titles]

    # Split the document content by these titles
    sections_content = re.split(section_pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
    sections_content = [
        content.strip() for content in sections_content
        if content.strip() and not content.strip().lower().startswith('item ')
    ]

    print(f"Identified {len(section_titles)} document sections.")

    # Handle mismatch between titles and content
    min_len = min(len(section_titles), len(sections_content))
    if len(section_titles) != len(sections_content):
        print(f"Warning: Mismatch between titles ({len(section_titles)}) and content ({len(sections_content)})")

    doc_chunks_with_metadata = []
    for i in range(min_len):
        section_title = section_titles[i]
        content = sections_content[i]

        # Chunk the content of this specific section
        section_chunks = text_splitter.split_text(content)
        for chunk in section_chunks:
            chunk_id = str(uuid.uuid4())
            doc_chunks_with_metadata.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "section": section_title,
                        "source_doc": source_path,
                        "id": chunk_id
                    }
                )
            )

    print(f"Created {len(doc_chunks_with_metadata)} chunks with section metadata.")
    return doc_chunks_with_metadata


def format_docs(docs: list[Document]) -> str:
    """
    Format a list of documents into a single string for LLM context.

    Args:
        docs: List of Document objects to format.

    Returns:
        A formatted string with documents separated by dividers.
    """
    if not docs:
        return ""

    formatted_parts = []
    for doc in docs:
        if hasattr(doc, 'page_content'):
            formatted_parts.append(doc.page_content)
        elif isinstance(doc, dict) and 'content' in doc:
            formatted_parts.append(doc['content'])
        elif isinstance(doc, dict) and 'context' in doc:
            formatted_parts.append(doc['context'])

    return "\n\n---\n\n".join(formatted_parts)
