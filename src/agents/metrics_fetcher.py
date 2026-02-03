"""
Stock metrics fetcher for the Deep Thinking RAG pipeline.

This module implements functions to fetch stock fundamentals and
analyst opinions from the Finnhub API.
"""

import os
import time
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from langchain_core.documents import Document


BASE_URL = "https://finnhub.io/api/v1"
MAX_RETRIES = 3
TIMEOUT = 5


def _finnhub_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a robust HTTP GET request to the Finnhub API.

    Implements retry logic with exponential backoff for reliability.

    Args:
        endpoint: The API endpoint path.
        params: Query parameters including the API token.

    Returns:
        The JSON response from the API.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                f"{BASE_URL}{endpoint}",
                params=params,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Finnhub request failed after {MAX_RETRIES} attempts: {endpoint}"
                ) from e
            time.sleep(2 ** attempt)  # exponential backoff


def _rag_doc(
    *,
    symbol: str,
    doc_type: str,
    content: str,
    metric: str = "",
) -> Dict[str, Any]:
    """
    Create a standardized RAG document from stock data.

    Args:
        symbol: The stock ticker symbol.
        doc_type: The type of document (e.g., 'fundamental_valuation_pe').
        content: The human-readable content string.
        metric: The specific metric name.

    Returns:
        A dictionary representing the RAG document.
    """
    return {
        "type": doc_type,
        "symbol": symbol,
        "metric": metric,
        "source": "finnhub",
        "as_of": datetime.now(timezone.utc).date().isoformat(),
        "context": content,
    }


def fetch_fundamentals(symbol: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch fundamental metrics for a stock.

    Retrieves P/E ratio, EV/FCF, net profit margin, and gross margin.

    Args:
        symbol: The stock ticker symbol.
        api_key: The Finnhub API key.

    Returns:
        List of RAG document dictionaries.
    """
    data = _finnhub_get(
        "/stock/metric",
        {"symbol": symbol, "metric": "all", "token": api_key},
    )

    m = data.get("metric", {})
    docs = []

    if "peNormalizedAnnual" in m and m["peNormalizedAnnual"] is not None:
        docs.append(
            _rag_doc(
                symbol=symbol,
                metric="peNormalizedAnnual",
                doc_type="fundamental_valuation_pe",
                content=(
                    f"{symbol} trades at a normalized P/E of "
                    f"{m['peNormalizedAnnual']:.2f}."
                ),
            )
        )

    if "currentEv/freeCashFlowTTM" in m and m["currentEv/freeCashFlowTTM"] is not None:
        docs.append(
            _rag_doc(
                symbol=symbol,
                metric="currentEv/freeCashFlowTTM",
                doc_type="fundamental_valuation_fcf",
                content=(
                    f"{symbol} has a current EV / FCF TTM of "
                    f"{m['currentEv/freeCashFlowTTM']:.1f}."
                ),
            )
        )

    if "netProfitMarginAnnual" in m and m["netProfitMarginAnnual"] is not None:
        docs.append(
            _rag_doc(
                symbol=symbol,
                metric="netProfitMarginAnnual",
                doc_type="fundamental_profitability",
                content=(
                    f"{symbol} has an annual net profit margin of "
                    f"{m['netProfitMarginAnnual']:.1f}%."
                ),
            )
        )

    if "grossMarginAnnual" in m and m["grossMarginAnnual"] is not None:
        docs.append(
            _rag_doc(
                symbol=symbol,
                metric="grossMarginAnnual",
                doc_type="fundamental_profitability",
                content=(
                    f"{symbol} has an annual gross margin of "
                    f"{m['grossMarginAnnual']:.1f}%."
                ),
            )
        )

    return docs


def fetch_analyst_opinions(symbol: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Fetch analyst recommendations for a stock from the last 6 months.

    Args:
        symbol: The stock ticker symbol.
        api_key: The Finnhub API key.

    Returns:
        List of RAG document dictionaries.
    """
    recs = _finnhub_get(
        "/stock/recommendation",
        {"symbol": symbol, "token": api_key},
    )

    docs = []
    six_months_ago = datetime.now(timezone.utc) - timedelta(days=30 * 6)

    for r in recs:
        try:
            period_date = datetime.strptime(r["period"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if period_date < six_months_ago:
            continue

        docs.append(
            _rag_doc(
                symbol=symbol,
                metric="analyst_forecast",
                doc_type="analyst_recommendation",
                content=(
                    f"As of {r['period']}, analysts rate {symbol} as "
                    f"{r['buy']} Buy, {r['hold']} Hold, and {r['sell']} Sell."
                ),
            )
        )

    return docs


def fetch_stock_rag_documents(symbol: str, api_key: str = None) -> List[Dict[str, Any]]:
    """
    Fetch all available RAG documents for a stock.

    Combines fundamentals and analyst opinions into a single list.
    Partial failures do not break the pipeline.

    Args:
        symbol: The stock ticker symbol.
        api_key: The Finnhub API key. If None, reads from environment.

    Returns:
        List of RAG document dictionaries.
    """
    if api_key is None:
        api_key = os.environ.get("FINNHUB_API_KEY")
        if not api_key:
            return [{
                "type": "error",
                "symbol": symbol,
                "source": "finnhub",
                "context": "FINNHUB_API_KEY not set in environment.",
            }]

    docs: List[Dict[str, Any]] = []

    for fn in (fetch_fundamentals, fetch_analyst_opinions):
        try:
            docs.extend(fn(symbol, api_key))
        except Exception as e:
            docs.append(
                _rag_doc(
                    symbol=symbol,
                    doc_type="error",
                    content=f"Failed to fetch {fn.__name__}: {str(e)}",
                )
            )

    return docs


def metrics_to_documents(metrics: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert metric dictionaries to LangChain Document objects.

    Args:
        metrics: List of metric dictionaries from fetch_stock_rag_documents.

    Returns:
        List of Document objects.
    """
    documents = []
    for m in metrics:
        doc = Document(
            page_content=m.get("context", ""),
            metadata={
                "source": m.get("source", "finnhub"),
                "type": m.get("type", ""),
                "symbol": m.get("symbol", ""),
            }
        )
        documents.append(doc)
    return documents
