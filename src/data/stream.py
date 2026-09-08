"""Alpaca WebSocket market data stream."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from alpaca.data.live import StockDataStream

from src.config import AppConfig
from src.data.bars import _parse_feed
from src.strategy.indicators import Quote

logger = logging.getLogger(__name__)

OnBarHandler = Callable[[str, dict[str, Any]], Awaitable[None]]
OnQuoteHandler = Callable[[Quote], Awaitable[None]]


class MarketDataStream:
    RECONNECT_MAX_DELAY_SECONDS = 60

    def __init__(
        self,
        config: AppConfig,
        symbols: list[str],
        on_bar: OnBarHandler,
        on_quote: OnQuoteHandler | None = None,
    ) -> None:
        self.config = config
        self.symbols = list(symbols)
        self.on_bar = on_bar
        self.on_quote = on_quote
        self._latest_quotes: dict[str, Quote] = {}
        self._stream = StockDataStream(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            feed=_parse_feed(config.alpaca_data_feed),
        )
        # Register before _run_forever so the initial connect subscribe includes them.
        # Do not call Alpaca subscribe_* after the stream is running — that SDK path
        # uses run_coroutine_threadsafe(...).result() and deadlocks on this event loop.
        for symbol in self.symbols:
            self._register_handlers(symbol)
        self._install_reconnect_backoff()

    def _install_reconnect_backoff(self) -> None:
        """Add exponential backoff to the SDK's connect retries.

        alpaca-py's ``_run_forever`` catches a failed ``_start_ws`` and loops with
        ``finally: await asyncio.sleep(0)`` — no delay at all. Auth failures raise
        ValueError, and only "insufficient subscription" exits the loop, so anything
        else (notably "connection limit exceeded", which the free IEX feed returns
        when another connection still holds the single allowed slot) becomes a hot
        loop retrying several times a second, forever. That never gives Alpaca time
        to release the old slot, so a transient condition turns into a permanent
        outage that also floods the log. Sleeping before the exception propagates
        turns that hot loop into a backed-off one.
        """
        original = self._stream._start_ws
        failures = 0

        async def _start_ws_with_backoff() -> None:
            nonlocal failures
            try:
                await original()
            except Exception as e:
                failures += 1
                delay = min(2**failures, self.RECONNECT_MAX_DELAY_SECONDS)
                logger.warning(
                    "Data stream connect failed (attempt %d): %s — retrying in %ds",
                    failures,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
                raise
            failures = 0

        self._stream._start_ws = _start_ws_with_backoff

    def get_quote(self, symbol: str) -> Quote | None:
        return self._latest_quotes.get(symbol)

    def _register_handlers(self, symbol: str) -> None:
        self._stream._ensure_coroutine(self._handle_bar)
        self._stream._ensure_coroutine(self._handle_quote)
        self._stream._handlers["bars"][symbol] = self._handle_bar
        self._stream._handlers["quotes"][symbol] = self._handle_quote

    def _drop_handlers(self, symbol: str) -> None:
        self._stream._handlers["bars"].pop(symbol, None)
        self._stream._handlers["quotes"].pop(symbol, None)

    def _on_stream_loop(self) -> bool:
        """True when called from the same running loop that owns the Alpaca stream."""
        loop = getattr(self._stream, "_loop", None)
        if loop is None or not loop.is_running():
            return False
        try:
            return asyncio.get_running_loop() is loop
        except RuntimeError:
            return False

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe additional symbols without blocking the event loop."""
        added: list[str] = []
        for symbol in symbols:
            if symbol in self.symbols:
                continue
            try:
                self._register_handlers(symbol)
                self.symbols.append(symbol)
                added.append(symbol)
            except Exception:
                logger.warning("Failed to subscribe %s to stream", symbol, exc_info=True)

        if not added:
            return

        # Before connect, handlers alone are enough — _run_forever sends subscribe.
        if getattr(self._stream, "_running", False):
            try:
                await self._stream._send_subscribe_msg()
            except Exception:
                logger.warning(
                    "Failed to push live subscribe for %s",
                    added,
                    exc_info=True,
                )
                return

        logger.info("Subscribed new symbols to stream: %s", added)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe symbols without blocking the event loop."""
        removed = [symbol for symbol in symbols if symbol in self.symbols]
        if not removed:
            return

        try:
            if getattr(self._stream, "_running", False):
                await self._stream._send_unsubscribe_msg("bars", removed)
                await self._stream._send_unsubscribe_msg("quotes", removed)
            for symbol in removed:
                self._drop_handlers(symbol)
                self.symbols.remove(symbol)
                self._latest_quotes.pop(symbol, None)
        except Exception:
            logger.debug(
                "Failed to unsubscribe %s from stream",
                removed,
                exc_info=True,
            )
            return

        logger.info("Unsubscribed symbols from stream: %s", removed)

    async def _handle_bar(self, bar: Any) -> None:
        symbol = bar.symbol
        data = {
            "t": str(bar.timestamp),
            "o": float(bar.open),
            "h": float(bar.high),
            "l": float(bar.low),
            "c": float(bar.close),
            "v": float(bar.volume),
        }
        await self.on_bar(symbol, data)

    async def _handle_quote(self, quote: Any) -> None:
        q = Quote(
            symbol=quote.symbol,
            bid=float(quote.bid_price),
            ask=float(quote.ask_price),
            timestamp=str(quote.timestamp),
        )
        self._latest_quotes[quote.symbol] = q
        if self.on_quote:
            await self.on_quote(q)

    async def run(self) -> None:
        logger.info("Starting market data stream for %s", self.symbols)
        try:
            await self._stream._run_forever()
        except asyncio.CancelledError:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the websocket without deadlocking when called on the stream loop."""
        try:
            stream = self._stream
            if self._on_stream_loop():
                # Alpaca's stop() uses run_coroutine_threadsafe(...).result(), which
                # deadlocks when invoked from the stream's own event loop thread.
                stream._should_run = False
                if stream._stop_stream_queue.empty():
                    stream._stop_stream_queue.put_nowait({"should_stop": True})
                return
            stream.stop()
        except Exception:
            logger.debug("Stream stop raised", exc_info=True)
