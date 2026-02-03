"""
Web search functionality for the Deep Thinking RAG pipeline.

This module implements web search using the Tavily Search API to
retrieve external, up-to-date information.
"""

from typing import List

from langchain_core.documents import Document

try:
    from langchain_tavily import TavilySearch
    _USE_NEW_TAVILY = True
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
    _USE_NEW_TAVILY = False


# Global web search tool instance
_web_search_tool = None


def get_web_search_tool():
    """
    Get or initialize the web search tool.

    Returns:
        The Tavily search tool instance.
    """
    global _web_search_tool

    if _web_search_tool is None:
        if _USE_NEW_TAVILY:
            _web_search_tool = TavilySearch(max_results=3)
        else:
            _web_search_tool = TavilySearchResults(k=3)
        print("Web search tool (Tavily) initialized.")

    return _web_search_tool


def web_search(query: str) -> List[Document]:
    """
    Perform a web search and return results as Document objects.

    Args:
        query: The search query string.

    Returns:
        List of Document objects with web search results.
    """
    try:
        tool = get_web_search_tool()

        if _USE_NEW_TAVILY:
            results = tool.invoke(query)
        else:
            results = tool.invoke({"query": query})

        documents = []
        for res in results:
            if isinstance(res, dict):
                doc = Document(
                    page_content=res.get("content", ""),
                    metadata={"source": res.get("url", "")}
                )
            else:
                # Handle new Tavily format if different
                doc = Document(
                    page_content=getattr(res, "content", str(res)),
                    metadata={"source": getattr(res, "url", "")}
                )
            documents.append(doc)

        return documents
    except Exception as e:
        print(f"ERROR in web_search: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
