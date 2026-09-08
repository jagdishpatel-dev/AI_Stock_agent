"""Regression: live subscribe/unsubscribe must not deadlock the event loop."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.stream import MarketDataStream


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        alpaca_api_key="PK_TEST",
        alpaca_secret_key="SECRET",
        alpaca_data_feed="iex",
    )


@pytest.mark.asyncio
async def test_subscribe_while_running_awaits_send_without_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    send = AsyncMock()
    fake_stream = MagicMock()
    fake_stream._handlers = {"bars": {}, "quotes": {}}
    fake_stream._running = True
    fake_stream._loop = asyncio.get_running_loop()
    fake_stream._ensure_coroutine = MagicMock()
    fake_stream._send_subscribe_msg = send
    fake_stream._send_unsubscribe_msg = AsyncMock()
    fake_stream._should_run = True
    fake_stream._stop_stream_queue = MagicMock()
    fake_stream._stop_stream_queue.empty.return_value = True

    monkeypatch.setattr("src.data.stream.StockDataStream", lambda **kwargs: fake_stream)
    monkeypatch.setattr("src.data.stream._parse_feed", lambda feed: feed)

    async def on_bar(symbol: str, data: dict) -> None:
        return None

    stream = MarketDataStream(_config(), ["AAPL"], on_bar=on_bar)  # type: ignore[arg-type]

    # Must complete on the same loop (old Alpaca .result() path deadlocked here).
    await asyncio.wait_for(stream.subscribe(["RIVN"]), timeout=1.0)

    assert "RIVN" in stream.symbols
    assert "RIVN" in fake_stream._handlers["bars"]
    assert "RIVN" in fake_stream._handlers["quotes"]
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_unsubscribe_while_running_awaits_send(monkeypatch: pytest.MonkeyPatch) -> None:
    unsub = AsyncMock()
    fake_stream = MagicMock()
    fake_stream._handlers = {"bars": {}, "quotes": {}}
    fake_stream._running = True
    fake_stream._loop = asyncio.get_running_loop()
    fake_stream._ensure_coroutine = MagicMock()
    fake_stream._send_subscribe_msg = AsyncMock()
    fake_stream._send_unsubscribe_msg = unsub
    fake_stream._should_run = True
    fake_stream._stop_stream_queue = MagicMock()
    fake_stream._stop_stream_queue.empty.return_value = True

    monkeypatch.setattr("src.data.stream.StockDataStream", lambda **kwargs: fake_stream)
    monkeypatch.setattr("src.data.stream._parse_feed", lambda feed: feed)

    async def on_bar(symbol: str, data: dict) -> None:
        return None

    stream = MarketDataStream(_config(), ["AAPL", "RIVN"], on_bar=on_bar)  # type: ignore[arg-type]
    await asyncio.wait_for(stream.unsubscribe(["RIVN"]), timeout=1.0)

    assert "RIVN" not in stream.symbols
    assert "RIVN" not in fake_stream._handlers["bars"]
    assert unsub.await_count == 2  # bars + quotes


@pytest.mark.asyncio
async def test_stop_on_same_loop_does_not_call_blocking_sdk_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_stream = MagicMock()
    fake_stream._handlers = {"bars": {}, "quotes": {}}
    fake_stream._running = True
    fake_stream._loop = asyncio.get_running_loop()
    fake_stream._ensure_coroutine = MagicMock()
    fake_stream._should_run = True
    fake_stream._stop_stream_queue = MagicMock()
    fake_stream._stop_stream_queue.empty.return_value = True

    monkeypatch.setattr("src.data.stream.StockDataStream", lambda **kwargs: fake_stream)
    monkeypatch.setattr("src.data.stream._parse_feed", lambda feed: feed)

    async def on_bar(symbol: str, data: dict) -> None:
        return None

    stream = MarketDataStream(_config(), ["AAPL"], on_bar=on_bar)  # type: ignore[arg-type]
    stream.stop()

    fake_stream.stop.assert_not_called()
    assert fake_stream._should_run is False
    fake_stream._stop_stream_queue.put_nowait.assert_called_once()


def test_connect_failures_back_off_instead_of_hot_looping() -> None:
    """Regression: alpaca-py retries a failed _start_ws with `sleep(0)`.

    Its `_run_forever` only exits the retry loop for "insufficient subscription";
    every other auth ValueError — notably "connection limit exceeded" — falls
    through to `finally: await asyncio.sleep(0)`, retrying several times a second
    forever. That both floods the log and stops Alpaca from ever releasing the
    single free-tier connection slot. MarketDataStream must add its own backoff.
    """
    from src.config import AppConfig
    from src.data.stream import MarketDataStream

    stream = MarketDataStream(
        AppConfig(alpaca_api_key="k", alpaca_secret_key="s"), ["NVDA"], on_bar=AsyncMock()
    )
    stream.RECONNECT_MAX_DELAY_SECONDS = 4

    async def always_fail() -> None:
        raise ValueError("connection limit exceeded")

    stream._stream._start_ws = always_fail
    stream._install_reconnect_backoff()

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    async def drive() -> None:
        with patch("src.data.stream.asyncio.sleep", fake_sleep):
            for _ in range(4):
                try:
                    await stream._stream._start_ws()
                except ValueError:
                    pass

    asyncio.run(drive())

    # Every failed attempt waits, and the delay grows then caps.
    assert slept == [2, 4, 4, 4], slept


def test_successful_connect_resets_backoff() -> None:
    """A reconnect after a good connection starts from the short delay again."""
    from src.config import AppConfig
    from src.data.stream import MarketDataStream

    stream = MarketDataStream(
        AppConfig(alpaca_api_key="k", alpaca_secret_key="s"), ["NVDA"], on_bar=AsyncMock()
    )
    should_fail = {"value": True}

    async def flaky() -> None:
        if should_fail["value"]:
            raise ValueError("connection limit exceeded")

    stream._stream._start_ws = flaky
    stream._install_reconnect_backoff()

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    async def drive() -> None:
        with patch("src.data.stream.asyncio.sleep", fake_sleep):
            for _ in range(2):
                try:
                    await stream._stream._start_ws()
                except ValueError:
                    pass
            should_fail["value"] = False
            await stream._stream._start_ws()  # succeeds, resets the counter
            should_fail["value"] = True
            try:
                await stream._stream._start_ws()
            except ValueError:
                pass

    asyncio.run(drive())

    assert slept == [2, 4, 2], slept
