"""Fetch and parse Alpaca NewsClient responses into article objects."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from src.config import AppConfig

logger = logging.getLogger(__name__)

RISK_KEYWORDS = ("earnings", "fda", "guidance", "downgrade", "lawsuit", "sec ", "investigation")


def fetch_recent_news(config: AppConfig, symbols: list[str]) -> list[dict]:
    client = NewsClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
    )
    start = datetime.now(timezone.utc) - timedelta(hours=config.briefing.news_lookback_hours)
    request = NewsRequest(
        symbols=",".join(symbols),
        start=start,
        limit=config.briefing.news_limit,
        include_content=False,
    )
    try:
        result = client.get_news(request)
    except Exception:
        logger.exception("Failed to fetch recent news")
        return []

    items: list[dict] = []
    for article in extract_news_articles(result):
        items.append(article_to_headline_fields(article))
    return items


def keyword_flags(news: list[dict], symbols: list[str]) -> dict[str, list[str]]:
    flags: dict[str, list[str]] = {s: [] for s in symbols}
    for article in news:
        headline = (article.get("headline") or "").lower()
        matched = [kw for kw in RISK_KEYWORDS if kw in headline]
        if not matched:
            continue
        for sym in article.get("symbols", []):
            if sym in flags:
                flags[sym].extend(matched)
    return {sym: list(set(hits)) for sym, hits in flags.items() if hits}


def extract_news_articles(result: Any) -> list[Any]:
    """Normalize Alpaca get_news() return value to a list of News objects or dicts."""
    data_attr = getattr(result, "data", None)
    if isinstance(data_attr, dict) and "news" in data_attr:
        return list(data_attr.get("news") or [])
    if hasattr(result, "news"):
        return list(getattr(result, "news") or [])
    if isinstance(result, dict):
        return list(result.get("news") or [])
    return []


def article_symbols(article: Any) -> list[str]:
    if isinstance(article, dict):
        return list(article.get("symbols") or [])
    return list(getattr(article, "symbols", None) or [])


def article_headline(article: Any) -> str:
    if isinstance(article, dict):
        return article.get("headline", "") or ""
    return getattr(article, "headline", "") or ""


def article_to_headline_fields(article: Any) -> dict[str, Any]:
    """Convert a News model or raw dict into normalized headline fields."""
    if isinstance(article, dict):
        return {
            "symbols": list(article.get("symbols") or []),
            "headline": article.get("headline", ""),
            "source": article.get("source", ""),
            "created_at": str(article.get("created_at", "")),
        }
    return {
        "symbols": list(article.symbols) if getattr(article, "symbols", None) else [],
        "headline": getattr(article, "headline", ""),
        "source": getattr(article, "source", ""),
        "created_at": str(getattr(article, "created_at", "")),
    }
