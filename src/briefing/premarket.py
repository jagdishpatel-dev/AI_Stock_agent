"""Pre-market news briefing and symbol avoid list."""

from __future__ import annotations

import logging

from src.config import AppConfig
from src.data.news_parse import fetch_recent_news, keyword_flags
from src.llm.ollama_client import PremarketBriefing
from src.llm.router import LLMRouter

logger = logging.getLogger(__name__)


async def run_briefing(config: AppConfig, llm: LLMRouter, symbols: list[str] | None = None) -> PremarketBriefing:
    watchlist = symbols or config.symbols
    news = fetch_recent_news(config, watchlist)
    flags = keyword_flags(news, watchlist)
    context = {
        "symbols": watchlist,
        "news": news,
        "keyword_flags": flags,
    }

    if not config.briefing.enabled:
        return PremarketBriefing(avoid=[], caution=[], reason="briefing disabled")

    if not news:
        logger.warning("No news available for pre-market briefing")
        keyword_avoid = list(flags.keys())
        return PremarketBriefing(
            avoid=keyword_avoid,
            caution=[],
            reason="No news feed — keyword flags only",
        )

    return await llm.briefing_decision(context)
