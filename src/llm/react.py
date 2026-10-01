"""ReAct (Reason + Act) watchlist ranking.

The model starts from a compact metrics table, may call read-only tools to dig
into individual symbols, and finishes with a ranked list. It has no tools that
place orders or change state — the ranking only feeds the entry gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.llm.prompts import REACT_WATCHLIST_PROMPT, REACT_WATCHLIST_RULES_SCALP, REACT_WATCHLIST_RULES_SWING

logger = logging.getLogger(__name__)

LLMCall = Callable[[str], Awaitable[dict | None]]
Tool = Callable[[dict], Awaitable[Any]]


@dataclass
class ToolSpec:
    fn: Tool
    description: str


@dataclass
class ReactResult:
    ranked: list[str]
    reason: str
    steps: list[dict] = field(default_factory=list)
    forced_final: bool = False


class WatchlistReactAgent:
    def __init__(
        self,
        llm_call: LLMCall,
        tools: dict[str, ToolSpec],
        *,
        max_steps: int = 4,
        observation_chars: int = 1500,
        swing: bool = False,
    ) -> None:
        self.llm_call = llm_call
        self.tools = tools
        self.max_steps = max_steps
        self.observation_chars = observation_chars
        self.swing = swing

    def _prompt(self, seed: dict, transcript: list[dict], steps_left: int) -> str:
        tools_text = "\n".join(f"- {name}: {spec.description}" for name, spec in self.tools.items())
        if steps_left > 0:
            budget = f"You may call up to {steps_left} more tool(s) before answering."
        else:
            budget = 'You have NO tool calls left. Your next response MUST use action "final".'
        return REACT_WATCHLIST_PROMPT.format(
            rules=REACT_WATCHLIST_RULES_SWING if self.swing else REACT_WATCHLIST_RULES_SCALP,
            tools=tools_text,
            seed=json.dumps(seed, indent=1, default=str),
            transcript=json.dumps(transcript, indent=1, default=str) if transcript else "(none yet)",
            budget=budget,
        )

    def _truncate(self, value: Any) -> Any:
        text = json.dumps(value, default=str)
        if len(text) <= self.observation_chars:
            return value
        return text[: self.observation_chars] + "…(truncated)"

    async def run(self, seed: dict) -> ReactResult | None:
        symbols = list(seed.keys())
        transcript: list[dict] = []
        tool_calls = 0

        # max_steps tool calls plus one forced final answer.
        for _ in range(self.max_steps + 1):
            steps_left = self.max_steps - tool_calls
            step = await self.llm_call(self._prompt(seed, transcript, steps_left))
            if not isinstance(step, dict):
                logger.warning("ReAct watchlist: LLM returned no parsable step")
                return None

            thought = str(step.get("thought", ""))[:500]
            action = str(step.get("action", "")).strip()
            args = step.get("args") if isinstance(step.get("args"), dict) else {}

            if action == "final" or steps_left <= 0:
                if action != "final":
                    logger.info("ReAct watchlist: step budget exhausted, model ignored final instruction")
                    return None
                ranked = clean_ranking(args.get("ranked"), symbols)
                if not ranked:
                    logger.warning("ReAct watchlist: final answer had no valid symbols")
                    return None
                transcript.append({"thought": thought, "action": "final"})
                return ReactResult(
                    ranked=ranked,
                    reason=str(args.get("reason", ""))[:500],
                    steps=transcript,
                    forced_final=steps_left <= 0,
                )

            tool_calls += 1
            spec = self.tools.get(action)
            if spec is None:
                observation: Any = {"error": f"unknown action '{action}'"}
            else:
                try:
                    observation = await spec.fn(args)
                except Exception as e:
                    logger.warning("ReAct tool %s failed: %s", action, e)
                    observation = {"error": f"{type(e).__name__}: {e}"}
            transcript.append(
                {
                    "thought": thought,
                    "action": action,
                    "args": args,
                    "observation": self._truncate(observation),
                }
            )

        return None


def clean_ranking(ranked: Any, symbols: list[str]) -> list[str]:
    """Uppercase, dedupe, and drop anything not on the watchlist."""
    if not isinstance(ranked, list):
        return []
    allowed = {s.upper() for s in symbols}
    out: list[str] = []
    for sym in ranked:
        s = str(sym).strip().upper()
        if s in allowed and s not in out:
            out.append(s)
    return out


def compute_gate(ranked: list[str], bottom_n: int, exempt: set[str] | None = None) -> set[str]:
    """Bottom N explicitly ranked symbols, excluding exempt (held) ones.

    Symbols the model left out are not gated — no evidence against them. If the
    ranking is too short to leave anything tradable, nothing is gated.
    """
    if bottom_n <= 0 or len(ranked) <= bottom_n:
        return set()
    exempt = exempt or set()
    return {s for s in ranked[-bottom_n:] if s not in exempt}


def react_run_times(
    day: datetime,
    start: str,
    end: str,
    runs: int,
    first_offset_minutes: int,
) -> list[datetime]:
    """`runs` evenly spaced times in [start + offset, end), on `day`'s date and tz."""
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    first = day.replace(hour=sh, minute=sm, second=0, microsecond=0) + timedelta(minutes=first_offset_minutes)
    close = day.replace(hour=eh, minute=em, second=0, microsecond=0)
    if runs <= 0 or close <= first:
        return []
    step = (close - first) / runs
    return [first + step * i for i in range(runs)]


def due_run_index(now: datetime, times: list[datetime], done: set[int]) -> int | None:
    """Latest scheduled slot that has passed and hasn't run.

    Earlier missed slots (e.g. agent started late) are skipped, never replayed,
    so the day never exceeds len(times) runs.
    """
    passed = [i for i, t in enumerate(times) if t <= now]
    if not passed:
        return None
    latest = passed[-1]
    return None if latest in done else latest


def build_watchlist_tools(
    *,
    bar_manager: Any,
    positions: Any,
    risk: Any,
    avoided_symbols: set[str],
    rsi_period: int,
    fetch_news: Callable[[list[str]], list[dict]],
    fetch_daily: Callable[[str, int], list[dict]],
) -> dict[str, ToolSpec]:
    """Read-only tools backed by the agent's live state."""

    def _symbol(args: dict) -> str:
        sym = str(args.get("symbol", "")).strip().upper()
        if sym not in bar_manager.states:
            raise ValueError(f"{sym or '(missing)'} is not on today's watchlist")
        return sym

    async def get_intraday(args: dict) -> dict:
        sym = _symbol(args)
        state = bar_manager.states[sym]
        n = min(int(args.get("bars", 10)), 30)
        avg_vol = state.avg_volume()
        latest = state.bars[-1] if state.bars else None

        def _r(v: float | None, d: int = 3) -> float | None:
            return round(v, d) if v is not None else None

        return {
            "symbol": sym,
            "close": state.latest_close(),
            "rsi": _r(state.rsi(rsi_period), 2),
            "vwap_dev_pct": _r(state.vwap_deviation_pct()),
            "gap_pct_from_prev_close": _r(state.gap_pct_from_prev_close()),
            "ema10": _r(state.ema(10)),
            "ema20": _r(state.ema(20)),
            "atr": _r(state.atr()),
            "volume_ratio": _r(latest.volume / avg_vol, 2) if latest and avg_vol else None,
            "recent_closes": [round(b.close, 2) for b in list(state.bars)[-n:]],
        }

    async def get_daily(args: dict) -> dict:
        sym = _symbol(args)
        days = min(int(args.get("days", 5)), 10)
        closes = await asyncio.to_thread(fetch_daily, sym, days)
        return {"symbol": sym, "daily_closes": closes}

    async def get_news(args: dict) -> dict:
        sym = _symbol(args)
        items = await asyncio.to_thread(fetch_news, [sym])
        return {
            "symbol": sym,
            "headlines": [
                {"headline": a.get("headline"), "created_at": a.get("created_at")} for a in items[:5]
            ],
        }

    async def get_position(args: dict) -> dict:
        sym = _symbol(args)
        pos = await asyncio.to_thread(positions.get_position, sym)
        if pos is None:
            return {"symbol": sym, "held": False}
        return {
            "symbol": sym,
            "held": True,
            "qty": str(getattr(pos, "qty", "")),
            "avg_entry_price": str(getattr(pos, "avg_entry_price", "")),
            "unrealized_plpc": str(getattr(pos, "unrealized_plpc", "")),
        }

    async def get_risk_state(args: dict) -> dict:
        st = risk.state
        return {
            "kill_switch": st.kill_switch,
            "realized_pnl_today": st.realized_pnl_today,
            "open_symbols": sorted(st.open_symbols),
            "premarket_avoid_list": sorted(avoided_symbols),
        }

    return {
        "get_intraday": ToolSpec(
            get_intraday,
            'Live intraday indicators and recent closes. args: {"symbol": str, "bars": int<=30}',
        ),
        "get_daily": ToolSpec(get_daily, 'Recent daily closes (trend). args: {"symbol": str, "days": int<=10}'),
        "get_news": ToolSpec(get_news, 'Latest news headlines. args: {"symbol": str}'),
        "get_position": ToolSpec(get_position, 'Whether we already hold it and its P&L. args: {"symbol": str}'),
        "get_risk_state": ToolSpec(get_risk_state, "Kill switch, today's P&L, pre-market avoid list. args: {}"),
    }
