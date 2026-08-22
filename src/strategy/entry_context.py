"""Extra context for entry LLM review (daily price history, trends)."""

from __future__ import annotations

import logging

from src.data.yahoo_client import fetch_recent_daily_closes

logger = logging.getLogger(__name__)


def build_daily_trend_context(
    symbol: str,
    current_price: float,
    days: int = 5,
) -> dict:
    """Compute recent daily closes and simple trend flags for LLM entry review."""
    try:
        bars = fetch_recent_daily_closes(symbol, days=days)
    except Exception:
        logger.warning("Daily price context fetch failed for %s", symbol, exc_info=True)
        bars = []

    if not bars:
        return {"daily_price_history": [], "daily_trend": {"available": False}}

    closes = [float(b["close"]) for b in bars]
    first_close = closes[0]
    change_vs_period_pct = (
        ((current_price - first_close) / first_close) * 100 if first_close > 0 else None
    )

    red_streak = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] < closes[i - 1]:
            red_streak += 1
        else:
            break

    period_high = max(closes)
    period_low = min(closes)
    dist_from_low_pct = (
        ((current_price - period_low) / period_low) * 100 if period_low > 0 else None
    )
    dist_from_high_pct = (
        ((current_price - period_high) / period_high) * 100 if period_high > 0 else None
    )

    return {
        "daily_price_history": bars,
        "daily_trend": {
            "available": True,
            "days": len(bars),
            "change_vs_period_start_pct": round(change_vs_period_pct, 3)
            if change_vs_period_pct is not None
            else None,
            "consecutive_down_days": red_streak,
            "period_high": round(period_high, 3),
            "period_low": round(period_low, 3),
            "dist_from_period_low_pct": round(dist_from_low_pct, 3)
            if dist_from_low_pct is not None
            else None,
            "dist_from_period_high_pct": round(dist_from_high_pct, 3)
            if dist_from_high_pct is not None
            else None,
            "pullback_from_high": dist_from_high_pct is not None and dist_from_high_pct < 0,
        },
    }
