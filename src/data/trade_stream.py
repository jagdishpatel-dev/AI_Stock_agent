"""Alpaca trading stream for order fill updates."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from alpaca.trading.stream import TradingStream

from src.config import AppConfig

logger = logging.getLogger(__name__)

OnOrderUpdate = Callable[[str, dict], Awaitable[None]]


class OrderUpdateStream:
    RECONNECT_MAX_DELAY_SECONDS = 60

    def __init__(self, config: AppConfig, on_update: OnOrderUpdate) -> None:
        self.config = config
        self.on_update = on_update
        self._stream = TradingStream(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            paper="paper" in config.alpaca_base_url,
        )
        self._stream.subscribe_trade_updates(self._handle_update)
        self._install_reconnect_backoff()

    def _install_reconnect_backoff(self) -> None:
        """Add exponential backoff to the SDK's connect retries.

        Same hot loop as MarketDataStream, and the same fix. alpaca-py's trading
        ``_run_forever`` catches a failed ``_start_ws`` and loops with
        ``finally: await asyncio.sleep(0.01)``, so any connect failure that isn't
        a clean shutdown retries ~100 times a second forever. A brief DNS outage
        on the host produced 23 reconnect attempts in 0.8s, each with a full
        traceback — order fills flow through this stream, so it floods the log
        and burns CPU at exactly the wrong moment. Sleeping before the exception
        propagates turns that hot loop into a backed-off one.
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
                    "Trading stream connect failed (attempt %d): %s — retrying in %ds",
                    failures,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
                raise
            failures = 0

        self._stream._start_ws = _start_ws_with_backoff

    async def _handle_update(self, update: object) -> None:
        event = str(getattr(update, "event", ""))
        order = getattr(update, "order", None)
        if order is None:
            return
        data = {
            "order_id": str(order.id),
            "id": str(order.id),
            "symbol": str(order.symbol),
            "side": str(order.side.value if hasattr(order.side, "value") else order.side),
            "filled_qty": float(order.filled_qty or 0),
            "filled_avg_price": float(order.filled_avg_price or 0),
            "qty": float(order.qty or 0),
            "price": float(order.filled_avg_price or 0),
            "status": str(order.status),
        }
        if event.lower() in ("fill", "partial_fill"):
            # Forward the real event name — collapsing both to "fill" defeated
            # the `if event != "fill"` guard downstream, so every partial fill
            # tick (cumulative filled_qty) was journaled as its own trade.
            await self.on_update(event.lower(), data)
        logger.debug("Order update: %s %s", event, data)

    async def run(self) -> None:
        logger.info("Starting trading update stream")
        try:
            await self._stream._run_forever()
        except asyncio.CancelledError:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop without deadlocking when called on the stream's event loop."""
        try:
            stream = self._stream
            loop = getattr(stream, "_loop", None)
            if loop is not None and loop.is_running():
                try:
                    on_loop = asyncio.get_running_loop() is loop
                except RuntimeError:
                    on_loop = False
                if on_loop:
                    # TradingStream.stop() uses run_coroutine_threadsafe(...).result()
                    stream._should_run = False
                    stop_queue = getattr(stream, "_stop_stream_queue", None)
                    if stop_queue is not None and stop_queue.empty():
                        stop_queue.put_nowait({"should_stop": True})
                    return
            stream.stop()
        except Exception:
            logger.debug("Trading stream stop raised", exc_info=True)
