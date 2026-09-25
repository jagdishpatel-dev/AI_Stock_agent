"""Tests for LLM learning helpers (outcome cards) and the ReAct decision loop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from src.config import AppConfig, LLMConfig
from src.journal.logger import TradeJournal, TradeRecord
from src.llm.openrouter_client import OpenRouterClient
from src.llm.react_engine import run_react_loop
from src.llm.react_tools import ALL_READ_TOOLS, ToolContext, build_tool_dispatch
from src.llm.router import LLMRouter
from src.strategy.indicators import IndicatorState


def test_outcome_cards_summary(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    journal = TradeJournal(str(db))

    journal.log_signal("PLTR", "entry", "momentum_breakout", "approve", 0.85, "ok", rsi=54.0, vwap_dev=1.2)
    journal.log_trade(
        TradeRecord(symbol="PLTR", side="buy", qty=10, price=100.0, order_id="b1", reason="swing_entry_fill")
    )
    journal.log_trade(
        TradeRecord(
            symbol="PLTR",
            side="sell",
            qty=10,
            price=98.0,
            order_id="s1",
            pnl=-20.0,
            reason="dynamic_stop",
        )
    )

    journal.log_signal("NVDA", "entry", "momentum_breakout", "approve", 0.9, "ok", rsi=55.0, vwap_dev=1.1)
    journal.log_trade(
        TradeRecord(symbol="NVDA", side="buy", qty=5, price=200.0, order_id="b2", reason="swing_entry_fill")
    )
    journal.log_trade(
        TradeRecord(
            symbol="NVDA",
            side="sell",
            qty=5,
            price=210.0,
            order_id="s2",
            pnl=50.0,
            reason="trailing_stop",
        )
    )

    cards = journal.get_outcome_cards(symbol="PLTR", rsi=54.5, vwap_dev=1.15, limit=5)
    assert cards["cards"]
    assert "W/" in cards["summary"] and "avg_pnl=$" in cards["summary"]
    assert any(c["symbol"] == "PLTR" and c["result"] == "loss" for c in cards["cards"])


def _make_tool_ctx(*, days_held: int = 1) -> ToolContext:
    return ToolContext(
        symbol="PLTR",
        config=AppConfig(),
        indicator_state=IndicatorState(),
        stream=Mock(get_quote=Mock(return_value=None)),
        positions=Mock(),
        journal=Mock(),
        swing=True,
        entry_price=100.0,
        entry_time=None,
        highest_since_entry=105.0,
        position_qty=10.0,
        days_held=days_held,
    )


def _submit_decision_message(**kwargs: object) -> dict:
    return {
        "content": "thinking...",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "submit_decision", "arguments": json.dumps(kwargs)},
            }
        ],
    }


def test_swing_review_react_unavailable_holds() -> None:
    """No OpenRouter configured -> fail safe to hold, no trade ever placed."""
    config = LLMConfig(enabled=True, confidence_threshold=0.7)
    router = LLMRouter(config)

    decision, trace = asyncio.run(router.swing_review_react(_make_tool_ctx()))

    assert decision.action == "hold"
    assert decision.confidence == 0.0
    assert trace == []


def test_swing_review_react_low_confidence_overridden_to_hold() -> None:
    """A low-confidence sell/trail from the model is never trusted — forced to hold."""
    config = LLMConfig(enabled=True, confidence_threshold=0.7, openrouter_api_key="k", openrouter_model="m")
    router = LLMRouter(config)
    router.openrouter.chat_with_tools = AsyncMock(  # type: ignore[method-assign]
        return_value=_submit_decision_message(
            action="sell",
            confidence=0.65,
            reasoning="High volume ratio and significant gap indicate a strong catalyst",
        )
    )

    decision, _ = asyncio.run(router.swing_review_react(_make_tool_ctx()))

    assert decision.action == "hold"
    assert decision.reasoning.startswith("low_confidence_hold:")


def test_run_react_loop_iteration_cap_fails_safe() -> None:
    """If submit_decision is never called, the loop fails safe instead of hanging or failing open."""
    openrouter = OpenRouterClient(LLMConfig(openrouter_api_key="k", openrouter_model="m"))
    read_only_message = {
        "content": "looking things up",
        "tool_calls": [
            {
                "id": "call_x",
                "type": "function",
                "function": {"name": "get_price_and_indicators", "arguments": "{}"},
            }
        ],
    }
    openrouter.chat_with_tools = AsyncMock(return_value=read_only_message)  # type: ignore[method-assign]

    tool_ctx = _make_tool_ctx()
    decision, trace = asyncio.run(
        run_react_loop(
            openrouter,
            system_prompt="system",
            user_prompt="user",
            read_tools=ALL_READ_TOOLS,
            tool_dispatch=build_tool_dispatch(tool_ctx),
            allowed_actions={"sell", "hold", "trail"},
            fail_safe_action="hold",
            max_iterations=3,
        )
    )

    assert decision.action == "hold"
    assert decision.confidence == 0.0
    assert decision.reasoning == "iteration_cap_exhausted"
    assert openrouter.chat_with_tools.await_count == 3
    assert len(trace) == 3  # one get_price_and_indicators call logged per turn


def test_run_react_loop_invalid_action_is_rejected_then_retried() -> None:
    """An out-of-scope action gets fed back as a tool error instead of being accepted."""
    openrouter = OpenRouterClient(LLMConfig(openrouter_api_key="k", openrouter_model="m"))
    openrouter.chat_with_tools = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _submit_decision_message(action="buy", confidence=0.9, reasoning="not allowed here"),
            _submit_decision_message(action="hold", confidence=0.8, reasoning="second try"),
        ]
    )

    tool_ctx = _make_tool_ctx()
    decision, trace = asyncio.run(
        run_react_loop(
            openrouter,
            system_prompt="system",
            user_prompt="user",
            read_tools=ALL_READ_TOOLS,
            tool_dispatch=build_tool_dispatch(tool_ctx),
            allowed_actions={"sell", "hold", "trail"},
            fail_safe_action="hold",
            max_iterations=5,
        )
    )

    assert decision.action == "hold"
    assert decision.reasoning == "second try"
    assert openrouter.chat_with_tools.await_count == 2
    assert [t["action"] for t in trace] == ["submit_decision", "submit_decision"]


def test_run_react_loop_tool_call_then_submit_decision_trace_order() -> None:
    """A read-only tool call followed by submit_decision produces an ordered trace of both."""
    openrouter = OpenRouterClient(LLMConfig(openrouter_api_key="k", openrouter_model="m"))
    openrouter.chat_with_tools = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            {
                "content": "checking position first",
                "tool_calls": [
                    {
                        "id": "call_pos",
                        "type": "function",
                        "function": {"name": "get_position", "arguments": "{}"},
                    }
                ],
            },
            _submit_decision_message(action="hold", confidence=0.75, reasoning="looks fine"),
        ]
    )

    tool_ctx = _make_tool_ctx()
    decision, trace = asyncio.run(
        run_react_loop(
            openrouter,
            system_prompt="system",
            user_prompt="user",
            read_tools=ALL_READ_TOOLS,
            tool_dispatch=build_tool_dispatch(tool_ctx),
            allowed_actions={"sell", "hold", "trail"},
            fail_safe_action="hold",
            max_iterations=5,
        )
    )

    assert decision.action == "hold"
    assert decision.reasoning == "looks fine"
    assert [t["action"] for t in trace] == ["get_position", "submit_decision"]
    assert trace[0]["observation"]["symbol"] == "PLTR"
