"""Regression: the trading stream must back off between failed reconnects."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from src.config import AppConfig
from src.data.trade_stream import OrderUpdateStream


def _stream() -> OrderUpdateStream:
    return OrderUpdateStream(
        AppConfig(alpaca_api_key="k", alpaca_secret_key="s"), on_update=AsyncMock()
    )


def test_connect_failures_back_off_instead_of_hot_looping() -> None:
    """alpaca-py's trading `_run_forever` retries a failed `_start_ws` every 10ms.

    Its loop ends in `finally: await asyncio.sleep(0.01)`, so a connect failure
    that isn't a clean shutdown retries ~100x/second forever. A DNS blip on the
    VM on 2026-09-22 produced 23 attempts in 0.8s, each with a full traceback.
    OrderUpdateStream must add its own backoff.
    """
    stream = _stream()
    stream.RECONNECT_MAX_DELAY_SECONDS = 4

    async def always_fail() -> None:
        raise OSError("[Errno -3] Temporary failure in name resolution")

    stream._stream._start_ws = always_fail
    stream._install_reconnect_backoff()

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    async def drive() -> None:
        with patch("src.data.trade_stream.asyncio.sleep", fake_sleep):
            for _ in range(4):
                try:
                    await stream._stream._start_ws()
                except OSError:
                    pass

    asyncio.run(drive())

    # Every failed attempt waits, and the delay grows then caps.
    assert slept == [2, 4, 4, 4], slept


def test_successful_connect_resets_backoff() -> None:
    stream = _stream()
    stream.RECONNECT_MAX_DELAY_SECONDS = 60

    outcomes = [OSError("dns"), OSError("dns"), None, OSError("dns")]

    async def flaky() -> None:
        result = outcomes.pop(0)
        if result is not None:
            raise result

    stream._stream._start_ws = flaky
    stream._install_reconnect_backoff()

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    async def drive() -> None:
        with patch("src.data.trade_stream.asyncio.sleep", fake_sleep):
            for _ in range(2):
                try:
                    await stream._stream._start_ws()
                except OSError:
                    pass
            await stream._stream._start_ws()  # succeeds, resets the counter
            try:
                await stream._stream._start_ws()
            except OSError:
                pass

    asyncio.run(drive())

    # 2s, 4s, then back to 2s after the success rather than continuing to 8s.
    assert slept == [2, 4, 2], slept


def test_backoff_is_installed_by_default() -> None:
    """The wrapper must be wired up in __init__, not just available as a method."""
    stream = _stream()
    assert stream._stream._start_ws.__name__ == "_start_ws_with_backoff"
