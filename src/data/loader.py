"""
Document loading utilities for the Deep Thinking RAG pipeline.

This module handles loading documents from various sources including
SEC 10-K filings and text files.
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from src.config.settings import CONFIG


def download_and_parse_10k(url: str, doc_path_raw: str, doc_path_clean: str) -> None:
    """
    Download and parse an SEC 10-K filing from EDGAR.

    Downloads the HTML filing, removes tables, and extracts clean text content.
    The raw HTML is saved to doc_path_raw and cleaned text to doc_path_clean.

    Args:
        url: The SEC EDGAR URL of the 10-K filing.
        doc_path_raw: Path to save the raw HTML file.
        doc_path_clean: Path to save the cleaned text file.
    """
    # Check if we need to download
    if not os.path.exists(doc_path_raw):
        print(f"Downloading 10-K filing from {url}...")
        headers = {'User-Agent': 'Research Project research@example.com'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        with open(doc_path_raw, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Raw document saved to {doc_path_raw}")

    # Parse the HTML
    with open(doc_path_raw, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove tables, which are often noisy for text-based RAG
    for table in soup.find_all('table'):
        table.decompose()

    # Get clean text, attempting to preserve paragraph breaks
    text = ''
    for p in soup.find_all(['p', 'div', 'span']):
        text += p.get_text(strip=True) + '\n\n'

    # Clean up excessive newlines and whitespace
    clean_text = re.sub(r'\n{3,}', '\n\n', text).strip()
    clean_text = re.sub(r'\s{2,}', ' ', clean_text).strip()

    with open(doc_path_clean, 'w', encoding='utf-8') as f:
        f.write(clean_text)
    print(f"Cleaned text content extracted and saved to {doc_path_clean}")


def load_text_document(file_path: str) -> list[Document]:
    """
    Load a text document using LangChain's TextLoader.

    Args:
        file_path: Path to the text file to load.

    Returns:
        A list of Document objects containing the file content.
    """
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()


def get_default_10k_paths() -> tuple[str, str, str]:
    """
    Get the default paths for NVIDIA's 2023 10-K filing.

    Returns:
        A tuple of (url, raw_path, clean_path).
    """
    url = "https://www.sec.gov/Archives/edgar/data/1045810/000104581023000017/nvda-20230129.htm"
    raw_path = os.path.join(CONFIG["raw_data_dir"], "nvda_10k_2023_raw.html")
    clean_path = os.path.join(CONFIG["processed_data_dir"], "nvda_10k_2023_clean.txt")
    return url, raw_path, clean_path
