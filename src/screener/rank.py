"""Rank screener candidates via LLM with rule-based fallback."""

from __future__ import annotations

import logging

from src.config import AppConfig
from src.llm.ollama_client import ScreenerRanking
from src.llm.router import LLMRouter

logger = logging.getLogger(__name__)


def rule_based_rank(candidates: list[dict], slots: int) -> ScreenerRanking:
    """Score = volume * (1 + abs(gap_pct)/10), prefer moderate gaps."""
    scored: list[tuple[float, dict]] = []
    for c in candidates:
        gap = abs(c.get("gap_pct", c.get("percent_change", 0)))
        if gap > 8:
            continue
        vol = c.get("volume", 0)
        score = vol * (1 + gap / 10)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    picks = [c["symbol"] for _, c in scored[:slots]]
    reasons = {
        c["symbol"]: f"vol={c.get('volume', 0):.0f} gap={c.get('gap_pct', 0)}%"
        for _, c in scored[:slots]
    }
    return ScreenerRanking(picks=picks, reasons=reasons, summary="rule-based fallback ranking")


async def rank_candidates(
    candidates: list[dict],
    config: AppConfig,
    llm: LLMRouter,
) -> ScreenerRanking:
    slots = config.screener.dynamic_slots
    if not candidates:
        return ScreenerRanking(picks=[], reasons={}, summary="no candidates")

    if config.llm.enabled and len(candidates) > slots:
        ranking = await llm.screener_rank(
            {"candidates": candidates, "slots": slots},
            swing=config.strategy.mode == "swing",
        )
        if ranking and ranking.picks:
            valid = [p for p in ranking.picks if p in {c["symbol"] for c in candidates}]
            if len(valid) >= min(slots, len(candidates)):
                ranking.picks = valid[:slots]
                return ranking

    return rule_based_rank(candidates, slots)
