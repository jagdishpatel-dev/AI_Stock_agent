"""Tests for LLM learning helpers: outcome cards + low-confidence swing hold."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.config import LLMConfig
from src.journal.logger import TradeJournal, TradeRecord
from src.llm.ollama_client import SwingReviewDecision
from src.llm.router import LLMRouter


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


def test_swing_review_low_confidence_holds() -> None:
    config = LLMConfig(enabled=True, confidence_threshold=0.7, primary_provider="ollama")
    router = LLMRouter(config)

    async def _fake_first_result(fns):
        return SwingReviewDecision(
            action="exit",
            confidence=0.65,
            reason="High volume ratio and significant gap indicate a strong catalyst",
        ), "google"

    router._first_result = _fake_first_result  # type: ignore[method-assign]
    router._ollama_healthy = False

    decision, source = asyncio.run(router.swing_review({"days_held": 1}))
    assert source == "google"
    assert decision.action == "hold"
    assert decision.reason.startswith("low_confidence_hold:")


def test_swing_review_unavailable_holds() -> None:
    config = LLMConfig(enabled=True, confidence_threshold=0.7, primary_provider="ollama")
    router = LLMRouter(config)

    async def _fake_first_result(fns):
        return None, "none"

    router._first_result = _fake_first_result  # type: ignore[method-assign]
    router._ollama_healthy = False

    decision, source = asyncio.run(router.swing_review({"days_held": 1}))
    assert source == "fail_safe"
    assert decision.action == "hold"
