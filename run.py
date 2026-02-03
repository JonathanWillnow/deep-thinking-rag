#!/usr/bin/env python
"""
Main entry point for running the Deep Thinking RAG pipeline.

This script provides a CLI interface to run complex queries through
the advanced agentic RAG system.

Usage:
    python run.py
    python run.py --query "Your complex question here"
    python run.py --baseline  # Run baseline comparison
"""

import argparse
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.markdown import Markdown

from src.config.settings import setup_environment, CONFIG
from src.data.loader import load_text_document, get_default_10k_paths, download_and_parse_10k
from src.data.processor import split_documents, extract_sections_with_metadata
from src.retrieval.vector_store import create_vector_store, initialize_advanced_vector_store
from src.retrieval.bm25 import initialize_bm25_index
from src.pipelines.deep_rag import run_deep_thinking_rag, display_final_answer, get_answer
from src.pipelines.shallow_rag import create_baseline_rag_chain, run_baseline_query


console = Console()

# Default complex query for demonstration
DEFAULT_QUERY = (
    "Based on NVIDIA's 2023 10-K filing, identify their key risks related to competition. "
    "Then, find recent news (post-filing, from 2024) about AMD's AI chip strategy and "
    "explain how this new strategy directly addresses or exacerbates one of NVIDIA's stated risks."
)


def initialize_knowledge_base():
    """
    Initialize the knowledge base from the 10-K document.

    Downloads the document if necessary, processes it into chunks,
    and initializes both vector and BM25 indexes.
    """
    console.print("\n[bold cyan]Initializing Knowledge Base...[/bold cyan]\n")

    # Get paths and download if needed
    url, raw_path, clean_path = get_default_10k_paths()

    # Ensure data directories exist
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)

    # Check if we need to download/process
    if not os.path.exists(clean_path):
        # Check if raw file exists in the old location
        old_raw_path = os.path.join(CONFIG["data_dir"], "nvda_10k_2023_raw.html")
        old_clean_path = os.path.join(CONFIG["data_dir"], "nvda_10k_2023_clean.txt")

        if os.path.exists(old_clean_path):
            # Use existing file from old location
            clean_path = old_clean_path
            console.print(f"Using existing document: {clean_path}")
        elif os.path.exists(old_raw_path):
            # Process existing raw file
            download_and_parse_10k(url, old_raw_path, old_clean_path)
            clean_path = old_clean_path
        else:
            console.print("[yellow]Document not found. Please ensure nvda_10k_2023_clean.txt exists in data/ folder.[/yellow]")
            console.print("You can download it from SEC EDGAR and process it manually.")
            return None, None

    # Load and process documents
    console.print("Loading document...")
    documents = load_text_document(clean_path)
    console.print(f"Document loaded: {len(documents[0].page_content)} characters")

    # Create chunks with metadata
    console.print("Extracting sections and creating chunks...")
    doc_chunks = extract_sections_with_metadata(documents, clean_path)

    # Initialize vector store
    console.print("Initializing vector store...")
    try:
        initialize_advanced_vector_store(doc_chunks)
        console.print("[green]Vector store initialized successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error initializing vector store: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None, None

    # Initialize BM25 index
    console.print("Initializing BM25 index...")
    initialize_bm25_index(doc_chunks)

    # Create baseline retriever
    console.print("Creating baseline retriever...")
    basic_chunks = split_documents(documents)
    baseline_vector_store = create_vector_store(basic_chunks)
    baseline_retriever = baseline_vector_store.as_retriever(search_kwargs={"k": 3})

    console.print("[bold green]Knowledge base initialized successfully![/bold green]\n")

    return doc_chunks, baseline_retriever


def run_query(query: str, show_baseline: bool = False, baseline_retriever=None):
    """
    Run a query through the Deep Thinking RAG pipeline.

    Args:
        query: The user's complex query.
        show_baseline: Whether to also run and compare with baseline.
        baseline_retriever: The baseline retriever for comparison.
    """
    console.print(f"\n[bold]Query:[/bold] {query}\n")
    console.print("=" * 80)

    # Run Deep Thinking RAG
    final_state = run_deep_thinking_rag(query, stream=True)

    # Display results
    display_final_answer(final_state)

    # Optionally run baseline comparison
    if show_baseline and baseline_retriever:
        console.print("\n" + "=" * 80)
        console.print("\n[bold yellow]BASELINE RAG (for comparison)[/bold yellow]\n")

        baseline_chain = create_baseline_rag_chain(baseline_retriever)
        baseline_answer = run_baseline_query(baseline_chain, query)

        console.print(Markdown(baseline_answer))

    return final_state


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run the Deep Thinking RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="The query to run. If not provided, uses the default complex query."
    )
    parser.add_argument(
        "--baseline", "-b",
        action="store_true",
        help="Also run baseline RAG for comparison."
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output."
    )

    args = parser.parse_args()

    # Setup environment
    console.print("[bold]Deep Thinking RAG Pipeline[/bold]")
    console.print("=" * 80)
    setup_environment()

    # Initialize knowledge base
    doc_chunks, baseline_retriever = initialize_knowledge_base()

    if doc_chunks is None:
        console.print("[red]Failed to initialize knowledge base. Exiting.[/red]")
        return 1

    # Get query
    query = args.query if args.query else DEFAULT_QUERY

    # Run query
    final_state = run_query(
        query,
        show_baseline=args.baseline,
        baseline_retriever=baseline_retriever
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        console.print(f"[red]Unhandled error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
