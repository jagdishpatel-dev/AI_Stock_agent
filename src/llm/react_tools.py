"""Read-only tools + the submit_decision tool for the ReAct trading-decision loop.

Every tool here is wired to the same live indicator state, Alpaca clients, and
journal already used by the rest of the agent — none of this is placeholder data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config import AppConfig
from src.data.news_parse import fetch_recent_news, keyword_flags
from src.data.stream import MarketDataStream
from src.execution.positions import PositionManager
from src.journal.logger import TradeJournal
from src.strategy.entry_context import build_daily_trend_context
from src.strategy.indicators import IndicatorState


@dataclass
class ToolContext:
    """Everything a tool implementation needs, bundled by the call site."""

    symbol: str
    config: AppConfig
    indicator_state: IndicatorState
    stream: MarketDataStream
    positions: PositionManager
    journal: TradeJournal
    swing: bool = False
    # Local position tracking — None at entry-veto time, populated once a
    # scalper/swing position is open (exit-advisor / swing-review call sites).
    entry_price: float | None = None
    entry_time: datetime | None = None
    highest_since_entry: float | None = None
    position_qty: float | None = None
    days_held: int | None = None


class ReactDecision(BaseModel):
    action: Literal["buy", "sell", "hold", "trail"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    # Optional, only meaningful for specific sites/actions:
    target_pct: float | None = None  # exit-advisor "hold": re-check target
    max_hold_minutes: int | None = None  # exit-advisor "hold": re-check timeout
    new_stop_pct: float | None = None  # swing-review "trail": tighter stop distance


SUBMIT_DECISION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": (
            "Submit your final trading decision. Call this exactly once, as your "
            "last action, only after you have gathered the information you need."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["buy", "sell", "hold", "trail"],
                    "description": "Only use an action that is valid for this decision — see system prompt.",
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string", "description": "Brief reason for the decision."},
                "target_pct": {
                    "type": "number",
                    "description": "Only for a 'hold' exit decision: percent gain to re-check at.",
                },
                "max_hold_minutes": {
                    "type": "integer",
                    "description": "Only for a 'hold' exit decision: minutes to hold before re-checking.",
                },
                "new_stop_pct": {
                    "type": "number",
                    "description": "Only for a 'trail' decision: new trailing stop distance (%) below the recent high.",
                },
            },
            "required": ["action", "confidence", "reasoning"],
        },
    },
}

GET_PRICE_AND_INDICATORS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_price_and_indicators",
        "description": "Get the latest live price, spread, and technical indicators for the symbol.",
        "parameters": {"type": "object", "properties": {}},
    },
}

GET_POSITION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_position",
        "description": "Get the current position for the symbol, if any (qty, entry price, P&L).",
        "parameters": {"type": "object", "properties": {}},
    },
}

GET_RECENT_NEWS_SENTIMENT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_recent_news_sentiment",
        "description": (
            "Get recent headlines for the symbol and any matched risk keywords "
            "(earnings, FDA, guidance, downgrade, lawsuit). Not a numeric sentiment score."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

GET_RECENT_TRADE_OUTCOMES_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_recent_trade_outcomes",
        "description": (
            "Get win/loss stats and outcome cards for recent trades on this symbol "
            "and similar RSI/VWAP setups from the trade journal."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

ALL_READ_TOOLS: list[dict] = [
    GET_PRICE_AND_INDICATORS_TOOL,
    GET_POSITION_TOOL,
    GET_RECENT_NEWS_SENTIMENT_TOOL,
    GET_RECENT_TRADE_OUTCOMES_TOOL,
]


async def tool_get_price_and_indicators(ctx: ToolContext) -> dict[str, Any]:
    ind = ctx.indicator_state
    strategy = ctx.config.strategy
    quote = ctx.stream.get_quote(ctx.symbol)
    close = ind.latest_close()
    avg_vol = ind.avg_volume()
    volume_ratio = (
        round(ind.bars[-1].volume / avg_vol, 2) if ind.bars and avg_vol else None
    )
    result: dict[str, Any] = {
        "symbol": ctx.symbol,
        "close": round(close, 4) if close is not None else None,
        "rsi": ind.rsi(strategy.rsi_period),
        "vwap_deviation_pct": ind.vwap_deviation_pct(),
        "volume_ratio": volume_ratio,
        "ema_fast": ind.ema(9),
        "ema_slow": ind.ema(21),
        "atr": ind.atr(),
        "gap_pct_from_prev_close": ind.gap_pct_from_prev_close(),
        "spread_pct": quote.spread_pct if quote else None,
    }
    if ctx.swing and close is not None:
        result.update(build_daily_trend_context(ctx.symbol, close))
    return result


async def tool_get_position(ctx: ToolContext) -> dict[str, Any]:
    if ctx.entry_price is None or not ctx.position_qty:
        pos = ctx.positions.get_position(ctx.symbol)
        if pos is None:
            return {"symbol": ctx.symbol, "has_position": False}
        return {
            "symbol": ctx.symbol,
            "has_position": True,
            "qty": float(pos.qty),
            "avg_entry_price": float(pos.avg_entry_price),
            "unrealized_plpc": round(float(pos.unrealized_plpc) * 100, 4),
        }

    close = ctx.indicator_state.latest_close() or ctx.entry_price
    pnl_pct = ((close - ctx.entry_price) / ctx.entry_price) * 100 if ctx.entry_price else 0.0
    return {
        "symbol": ctx.symbol,
        "has_position": True,
        "entry_price": round(ctx.entry_price, 4),
        "current_price": round(close, 4),
        "qty": ctx.position_qty,
        "pnl_pct": round(pnl_pct, 4),
        "highest_since_entry": round(ctx.highest_since_entry, 4) if ctx.highest_since_entry else None,
        "days_held": ctx.days_held,
    }


async def tool_get_recent_news_sentiment(ctx: ToolContext) -> dict[str, Any]:
    news = fetch_recent_news(ctx.config, [ctx.symbol])
    flags = keyword_flags(news, [ctx.symbol])
    headlines = [n.get("headline", "") for n in news if ctx.symbol in n.get("symbols", [])]
    return {
        "symbol": ctx.symbol,
        "headlines": headlines[:10],
        "risk_keyword_flags": flags.get(ctx.symbol, []),
        "note": "keyword-based risk flags, not a machine sentiment score",
    }


async def tool_get_recent_trade_outcomes(ctx: ToolContext) -> dict[str, Any]:
    jc = ctx.config.journal_context
    rsi = ctx.indicator_state.rsi(ctx.config.strategy.rsi_period)
    vwap_dev = ctx.indicator_state.vwap_deviation_pct()
    stats = ctx.journal.get_similar_setup_stats(
        ctx.symbol,
        rsi or 0.0,
        vwap_dev or 0.0,
        rsi_tolerance=jc.rsi_tolerance,
        lookback_days=jc.lookback_days,
    )
    cards = ctx.journal.get_outcome_cards(
        symbol=ctx.symbol,
        rsi=rsi,
        vwap_dev=vwap_dev,
        rsi_tolerance=jc.rsi_tolerance,
        vwap_tolerance=jc.outcome_cards_vwap_tolerance,
        lookback_days=jc.lookback_days,
        limit=jc.outcome_cards_limit,
    )
    return {"similar_setup_stats": stats, "outcome_cards": cards}


def build_tool_dispatch(ctx: ToolContext) -> dict[str, Any]:
    """Map tool name -> zero-arg async callable bound to this ToolContext."""
    return {
        "get_price_and_indicators": lambda: tool_get_price_and_indicators(ctx),
        "get_position": lambda: tool_get_position(ctx),
        "get_recent_news_sentiment": lambda: tool_get_recent_news_sentiment(ctx),
        "get_recent_trade_outcomes": lambda: tool_get_recent_trade_outcomes(ctx),
    }
