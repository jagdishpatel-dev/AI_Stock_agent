"""ReAct watchlist ranking: loop, guardrails, gate, and daily schedule."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytz

from src.llm.react import (
    ToolSpec,
    WatchlistReactAgent,
    clean_ranking,
    compute_gate,
    due_run_index,
    react_run_times,
)

SEED = {"AAPL": {"close": 1}, "MSFT": {"close": 1}, "NVDA": {"close": 1}, "SPY": {"close": 1}}


def _scripted(steps: list[dict | None]):
    prompts: list[str] = []

    async def call(prompt: str) -> dict | None:
        prompts.append(prompt)
        return steps.pop(0) if steps else None

    return call, prompts


def _tools(calls: list[dict]) -> dict[str, ToolSpec]:
    async def get_news(args: dict) -> dict:
        calls.append(args)
        return {"headlines": ["NVDA beats"]}

    async def broken(args: dict) -> dict:
        raise RuntimeError("feed down")

    return {"get_news": ToolSpec(get_news, "news"), "broken": ToolSpec(broken, "always fails")}


def test_tool_call_then_final_ranking() -> None:
    calls: list[dict] = []
    llm, prompts = _scripted(
        [
            {"thought": "check news", "action": "get_news", "args": {"symbol": "NVDA"}},
            {"thought": "done", "action": "final", "args": {"ranked": ["nvda", "AAPL", "MSFT", "SPY"], "reason": "r"}},
        ]
    )
    result = asyncio.run(WatchlistReactAgent(llm, _tools(calls)).run(SEED))

    assert result is not None
    assert result.ranked == ["NVDA", "AAPL", "MSFT", "SPY"]
    assert calls == [{"symbol": "NVDA"}]
    assert result.steps[0]["observation"] == {"headlines": ["NVDA beats"]}
    assert "NVDA beats" in prompts[1]  # observation fed back to the model


def test_unknown_and_failing_tools_become_observations() -> None:
    llm, _ = _scripted(
        [
            {"action": "place_order", "args": {"symbol": "NVDA"}},
            {"action": "broken", "args": {}},
            {"action": "final", "args": {"ranked": ["SPY"], "reason": ""}},
        ]
    )
    result = asyncio.run(WatchlistReactAgent(llm, _tools([])).run(SEED))

    assert result is not None
    assert "unknown action" in result.steps[0]["observation"]["error"]
    assert "feed down" in result.steps[1]["observation"]["error"]


def test_step_budget_caps_llm_calls() -> None:
    loop_forever = [{"action": "get_news", "args": {"symbol": "AAPL"}}] * 20
    llm, prompts = _scripted(list(loop_forever))
    result = asyncio.run(WatchlistReactAgent(llm, _tools([]), max_steps=4).run(SEED))

    assert result is None  # never answered final -> caller falls back to one-shot
    assert len(prompts) == 5  # 4 tool calls + 1 forced-final attempt
    assert "MUST use action \"final\"" in prompts[-1]


def test_unparsable_step_returns_none() -> None:
    llm, _ = _scripted([None])
    assert asyncio.run(WatchlistReactAgent(llm, _tools([])).run(SEED)) is None


def test_clean_ranking_drops_unknown_and_duplicates() -> None:
    assert clean_ranking(["aapl", "TSLA", "AAPL", "spy"], list(SEED)) == ["AAPL", "SPY"]
    assert clean_ranking("AAPL", list(SEED)) == []


def test_compute_gate() -> None:
    ranked = ["A", "B", "C", "D", "E"]
    assert compute_gate(ranked, 2) == {"D", "E"}
    assert compute_gate(ranked, 2, exempt={"E"}) == {"D"}
    assert compute_gate(ranked, 0) == set()
    assert compute_gate(["A", "B"], 3) == set()  # too short: gate nothing


def test_five_runs_evenly_spaced_across_session() -> None:
    tz = pytz.timezone("America/Chicago")
    day = tz.localize(datetime(2026, 10, 1, 7, 0))
    times = react_run_times(day, "08:30", "15:00", 5, 15)

    assert [t.strftime("%H:%M") for t in times] == ["08:45", "10:00", "11:15", "12:30", "13:45"]


def test_late_start_skips_missed_slots_instead_of_replaying() -> None:
    tz = pytz.timezone("America/Chicago")
    day = tz.localize(datetime(2026, 10, 1, 7, 0))
    times = react_run_times(day, "08:30", "15:00", 5, 15)

    assert due_run_index(day.replace(hour=8, minute=40), times, set()) is None
    # Agent comes up at 11:20: only the 11:15 slot runs; 08:45 and 10:00 are skipped.
    assert due_run_index(day.replace(hour=11, minute=20), times, set()) == 2
    assert due_run_index(day.replace(hour=11, minute=20), times, {0, 1, 2}) is None
