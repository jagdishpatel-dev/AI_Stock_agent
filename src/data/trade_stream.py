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
    def __init__(self, config: AppConfig, on_update: OnOrderUpdate) -> None:
        self.config = config
        self.on_update = on_update
        self._stream = TradingStream(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            paper="paper" in config.alpaca_base_url,
        )
        self._stream.subscribe_trade_updates(self._handle_update)

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
