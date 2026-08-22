"""Build LLM context for AI exit advisor decisions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytz

from src.config import AIExitConfig, AppConfig, SessionConfig
from src.llm.react_tools import ReactDecision
from src.strategy.indicators import IndicatorState


def session_minutes_left(session: SessionConfig) -> int:
    tz = pytz.timezone(session.timezone)
    now = datetime.now(tz)
    if now.weekday() >= 5:
        return 0
    end_h, end_m = map(int, session.end_time.split(":"))
    end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if now >= end:
        return 0
    return max(0, int((end - now).total_seconds() // 60))


def _trend_flags(bars: list) -> dict[str, object]:
    if len(bars) < 5:
        return {"higher_closes_last_5": False, "volume_trend": "unknown"}
    recent = bars[-5:]
    closes = [b.close for b in recent]
    higher = all(closes[i] >= closes[i - 1] for i in range(1, len(closes)))
    vol_first = sum(b.volume for b in recent[:2]) / 2
    vol_last = sum(b.volume for b in recent[-2:]) / 2
    if vol_last > vol_first * 1.1:
        vol_trend = "rising"
    elif vol_last < vol_first * 0.9:
        vol_trend = "falling"
    else:
        vol_trend = "flat"
    return {"higher_closes_last_5": higher, "volume_trend": vol_trend}


def build_exit_context(
    symbol: str,
    entry_price: float,
    entry_time: datetime,
    current_price: float,
    indicator_state: IndicatorState,
    config: AppConfig,
    zone: str,
) -> dict:
    ai = config.ai_exit
    strategy = config.strategy
    pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
    now = datetime.now(timezone.utc)
    minutes_in_trade = max(0, int((now - entry_time).total_seconds() // 60))

    bar_list = list(indicator_state.bars)
    n = ai.bar_context_count
    recent_bars = [
        {
            "t": b.timestamp,
            "o": round(b.open, 4),
            "h": round(b.high, 4),
            "l": round(b.low, 4),
            "c": round(b.close, 4),
            "v": round(b.volume, 0),
        }
        for b in bar_list[-n:]
    ]

    return {
        "zone": zone,
        "symbol": symbol,
        "entry_price": round(entry_price, 4),
        "current_price": round(current_price, 4),
        "pnl_pct": round(pnl_pct, 4),
        "minutes_in_trade": minutes_in_trade,
        "session_minutes_left": session_minutes_left(config.session),
        "rsi": indicator_state.rsi(strategy.rsi_period),
        "vwap_deviation_pct": indicator_state.vwap_deviation_pct(),
        "volume_ratio": (
            round(bar_list[-1].volume / indicator_state.avg_volume(), 2)
            if bar_list and indicator_state.avg_volume()
            else None
        ),
        "recent_bars": recent_bars,
        "trend": _trend_flags(bar_list),
        "_ai_exit_limits": {
            "hard_stop_loss_pct": ai.hard_stop_loss_pct,
            "min_take_profit_pct": ai.min_take_profit_pct,
            "max_target_pct": ai.max_target_pct,
            "max_hold_minutes": ai.max_hold_minutes,
            "max_loss_hold_minutes": ai.max_loss_hold_minutes,
        },
    }


def normalize_exit_decision(
    decision: ReactDecision,
    zone: str,
    ai_exit: AIExitConfig,
) -> ReactDecision:
    if decision.action == "sell":
        return decision

    if zone == "profit":
        target = decision.target_pct if decision.target_pct is not None else ai_exit.min_take_profit_pct
        target = max(ai_exit.min_take_profit_pct, min(target, ai_exit.max_target_pct))
        hold_mins = decision.max_hold_minutes or ai_exit.max_hold_minutes
        hold_mins = max(1, min(hold_mins, ai_exit.max_hold_minutes))
    else:
        target = decision.target_pct if decision.target_pct is not None else 0.0
        target = max(0.0, min(target, ai_exit.max_target_pct))
        hold_mins = decision.max_hold_minutes or ai_exit.max_loss_hold_minutes
        hold_mins = max(1, min(hold_mins, ai_exit.max_loss_hold_minutes))

    return ReactDecision(
        action="hold",
        target_pct=round(target, 4),
        max_hold_minutes=hold_mins,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
    )

