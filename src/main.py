"""Trading agent orchestrator."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.briefing.premarket import run_briefing
from src.config import AppConfig, load_config
from src.data.yahoo_client import fetch_finnhub_quote, fetch_quote_snapshot
from src.logging_sanitize import SensitiveDataFilter
from src.data.bars import BarManager
from src.data.stream import MarketDataStream
from src.data.trade_stream import OrderUpdateStream
from src.execution.hydration import held_symbols, hydrate_swing_positions
from src.execution.orders import OrderExecutor
from src.execution.positions import PositionManager
from src.journal.logger import TradeJournal
from src.llm.router import LLMRouter
from src.risk.manager import RiskManager
from src.screener.daily import build_daily_watchlist
from src.strategy.scalper import SymbolScalper
from src.strategy.swing_scalper import SwingScalper

_log_dir = Path(__file__).resolve().parent.parent / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

_log_filter = SensitiveDataFilter()
_handlers: list[logging.Handler] = [
    logging.StreamHandler(),
    logging.FileHandler(Path(__file__).resolve().parent.parent / "logs" / "agent.log"),
]
for _handler in _handlers:
    _handler.addFilter(_log_filter)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=_handlers,
)
logger = logging.getLogger(__name__)


class TradingAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.journal = TradeJournal(config.journal_db_path)
        self.executor = OrderExecutor(config)
        self.positions = PositionManager(config)
        self.risk = RiskManager(config, self.positions)
        self.llm = LLMRouter(config.llm)
        self.bar_manager = BarManager(config)
        self.scalpers: dict[str, SymbolScalper | SwingScalper] = {}
        self.active_symbols: list[str] = list(config.symbols)
        self.avoided_symbols: set[str] = set()
        self._avoid_list_date: str | None = None
        self._stream: MarketDataStream | None = None
        self._trade_stream: OrderUpdateStream | None = None
        self._running = False
        self._stopping = False
        self._watchlist_task: asyncio.Task | None = None
        self._session_task: asyncio.Task | None = None
        self._stream_task: asyncio.Task | None = None
        self._trade_stream_task: asyncio.Task | None = None
        self._morning_review_task: asyncio.Task | None = None
        self._daily_screener_task: asyncio.Task | None = None
        self._session_closed_date: str | None = None
        self._morning_reviewed_date: str | None = None
        self._screener_ran_date: str | None = None

    @property
    def _is_swing(self) -> bool:
        return self.config.strategy.mode == "swing"

    def _in_trading_session(self) -> bool:
        tz = pytz.timezone(self.config.session.timezone)
        now = datetime.now(tz)
        if now.weekday() >= 5:
            return False
        start_h, start_m = map(int, self.config.session.start_time.split(":"))
        if self._is_swing:
            end_time = self.config.swing.session_end_time
        else:
            end_time = self.config.session.end_time
        end_h, end_m = map(int, end_time.split(":"))
        start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        return start <= now <= end

    async def _wait_until_time(self, time_str: str) -> None:
        tz = pytz.timezone(self.config.session.timezone)
        now = datetime.now(tz)
        if now.weekday() >= 5:
            return
        h, m = map(int, time_str.split(":"))
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if now < target:
            wait_secs = (target - now).total_seconds()
            logger.info("Waiting %.0f seconds until %s CST", wait_secs, time_str)
            await asyncio.sleep(wait_secs)

    async def _run_screener(self) -> None:
        if not self.config.screener.enabled or self.config.screener.mode == "static":
            self.active_symbols = list(self.config.symbols)
            logger.info("Using static watchlist: %s", self.active_symbols)
            return

        await self._wait_until_time(self.config.screener.run_time)
        result = await build_daily_watchlist(self.config, self.llm, self.journal)
        self.active_symbols = result.symbols

        lines = [f"Today's watchlist: {', '.join(result.symbols)}"]
        for sym in result.dynamic_picks:
            reason = result.reasons.get(sym, "")
            lines.append(f"  {sym}: {reason}")
        lines.append(f"Summary: {result.summary}")
        msg = "\n".join(lines)
        logger.info("Screener result: %s", msg.replace("\n", " | "))
        await self.llm.alert("Daily Screener", msg)

    async def _wait_until_briefing_time(self) -> None:
        if not self.config.briefing.enabled:
            return
        await self._wait_until_time(self.config.briefing.time)

    async def _run_premarket_briefing(self) -> None:
        if not self.config.briefing.enabled:
            return
        await self._wait_until_briefing_time()
        briefing = await run_briefing(self.config, self.llm, self.active_symbols)
        tz = pytz.timezone(self.config.session.timezone)
        today = datetime.now(tz).date().isoformat()
        if self._avoid_list_date != today:
            self.avoided_symbols.clear()
            self._avoid_list_date = today
        new_avoid = {s.upper() for s in briefing.avoid}
        self.avoided_symbols |= new_avoid
        msg = (
            f"Avoid: {', '.join(sorted(self.avoided_symbols)) or 'none'}\n"
            f"Caution: {', '.join(briefing.caution) or 'none'}\n"
            f"Reason: {briefing.reason}"
        )
        logger.info("Pre-market briefing: %s", msg.replace("\n", " | "))
        self.journal.log_event("premarket_briefing", msg)
        await self.llm.alert("Pre-Market Briefing", msg)

    @staticmethod
    def _scalper_active(scalper: SymbolScalper | SwingScalper | None) -> bool:
        """True if a scalper is holding or working an order (not idle)."""
        if scalper is None:
            return False
        state = getattr(scalper, "state", None)
        value = getattr(state, "value", state)
        return str(value).lower() != "idle"

    def _symbol_has_exposure(self, symbol: str) -> bool:
        """True if the agent or Alpaca still has an open/working position."""
        if self._scalper_active(self.scalpers.get(symbol)):
            return True
        return self.positions.get_position_qty(symbol) > 0

    def _merge_held_symbols(self, symbols: list[str]) -> list[str]:
        """Ensure Alpaca-held symbols stay on the watchlist and data stream."""
        return list(dict.fromkeys([*symbols, *held_symbols(self.positions)]))

    async def _refresh_watchlist(self) -> None:
        """Re-run the screener mid-life and reconcile symbols, scalpers, and the stream."""
        if self._stream is None:
            return

        prev_symbols = list(self.active_symbols)
        await self._run_screener()
        await self._run_premarket_briefing()
        new_symbols = list(self.active_symbols)

        added = [s for s in new_symbols if s not in self.scalpers]
        removed = [s for s in prev_symbols if s not in new_symbols]
        # Never drop a symbol that still has an open/working position (agent or Alpaca).
        dropped = [s for s in removed if not self._symbol_has_exposure(s)]
        holdouts = [s for s in removed if s not in dropped]
        final_symbols = self._merge_held_symbols(list(dict.fromkeys(new_symbols + holdouts)))
        added = [s for s in final_symbols if s not in self.scalpers]
        self.active_symbols = final_symbols

        self.bar_manager.set_symbols(final_symbols)
        if added:
            await self.bar_manager.warmup_async(added)

        ScalperClass = SwingScalper if self._is_swing else SymbolScalper
        for symbol in added:
            self.scalpers[symbol] = ScalperClass(
                symbol,
                self.config,
                self.executor,
                self.risk,
                self.journal,
                self.llm,
                self._stream,
                avoided_symbols=self.avoided_symbols,
            )

        # Refresh the avoid list reference on every live scalper.
        for scalper in self.scalpers.values():
            scalper.avoided_symbols = self.avoided_symbols

        if added:
            await self._stream.subscribe(added)
        if dropped:
            await self._stream.unsubscribe(dropped)
            for symbol in dropped:
                self.scalpers.pop(symbol, None)

        if self._is_swing:
            await hydrate_swing_positions(
                self.positions,
                self.scalpers,
                self.journal,
                self.bar_manager,
            )

        logger.info(
            "Daily watchlist refreshed: %s (added=%s, dropped=%s)",
            final_symbols,
            added or "none",
            dropped or "none",
        )

    async def _daily_screener_loop(self) -> None:
        """Re-run the screener once per weekday at run_time so the watchlist stays fresh."""
        if not self.config.screener.enabled or self.config.screener.mode == "static":
            return

        tz = pytz.timezone(self.config.session.timezone)
        while self._running:
            await asyncio.sleep(60)
            now = datetime.now(tz)
            if now.weekday() >= 5:
                continue

            today = now.date().isoformat()
            if today == self._screener_ran_date:
                continue

            run_h, run_m = map(int, self.config.screener.run_time.split(":"))
            run_dt = now.replace(hour=run_h, minute=run_m, second=0, microsecond=0)
            if now < run_dt:
                continue

            self._screener_ran_date = today
            logger.info("Daily screener loop: re-running screener for %s", today)
            try:
                await self._refresh_watchlist()
            except Exception:
                logger.exception("Daily screener refresh failed")

    async def _on_bar(self, symbol: str, bar_data: dict) -> None:
        if not self._in_trading_session():
            return
        if self.risk.is_killed():
            return

        state = self.bar_manager.on_bar(symbol, bar_data)
        scalper = self.scalpers.get(symbol)
        if scalper:
            await scalper.on_bar(state)

    async def _watchlist_loop(self) -> None:
        interval = self.config.llm.watchlist_interval_minutes * 60
        while self._running:
            await asyncio.sleep(interval)
            if not self._in_trading_session() or not self.config.llm.enabled:
                continue

            symbols = list(self.bar_manager.states.keys())
            yahoo_snapshots: dict = {}
            if self.config.research.provider == "yahoo" and self.config.research.yahoo_enabled:
                try:
                    yahoo_snapshots = fetch_quote_snapshot(symbols)
                except Exception:
                    logger.warning("Yahoo watchlist snapshot failed", exc_info=True)

            context: dict = {}
            for symbol, state in self.bar_manager.states.items():
                rsi = state.rsi(self.config.strategy.rsi_period)
                vwap_dev = state.vwap_deviation_pct()
                close = state.latest_close()
                entry: dict = {
                    "rsi": rsi,
                    "vwap_dev": vwap_dev,
                    "close": close,
                }

                live_missing = rsi is None or vwap_dev is None or close is None
                snap = yahoo_snapshots.get(symbol)
                if live_missing and snap:
                    entry["research_price"] = snap.price
                    entry["research_change_pct"] = snap.change_pct
                    entry["research_volume"] = snap.volume
                    entry["metrics_source"] = snap.metrics_source
                    entry["data_quality"] = "partial"
                elif live_missing and self.config.research.finnhub_fallback and self.config.finnhub_api_key:
                    fh = fetch_finnhub_quote(symbol, self.config.finnhub_api_key)
                    if fh:
                        entry["research_price"] = fh.price
                        entry["research_change_pct"] = fh.change_pct
                        entry["metrics_source"] = fh.metrics_source
                        entry["data_quality"] = "partial"
                else:
                    entry["data_quality"] = "live"

                context[symbol] = entry

            ranking = await self.llm.rank_watchlist(
                context,
                swing=self.config.strategy.mode == "swing",
            )
            if ranking:
                logger.info("Watchlist ranking: %s — %s", ranking.ranked, ranking.reason)
                self.journal.log_event("watchlist_rank", str(ranking.ranked))

    async def _on_order_update(self, event: str, order_data: dict) -> None:
        symbol = order_data.get("symbol", "")
        scalper = self.scalpers.get(symbol)
        if scalper:
            await scalper.on_order_update(event, order_data)

    async def _morning_review_loop(self) -> None:
        """Swing mode: review every open position at morning_review_time each day."""
        tz = pytz.timezone(self.config.session.timezone)
        review_time = self.config.swing.morning_review_time

        while self._running:
            await asyncio.sleep(60)
            now = datetime.now(tz)
            if now.weekday() >= 5:
                continue
            today = now.date().isoformat()
            if today == self._morning_reviewed_date:
                continue

            review_h, review_m = map(int, review_time.split(":"))
            review_dt = now.replace(hour=review_h, minute=review_m, second=0, microsecond=0)
            if now < review_dt:
                continue

            self._morning_reviewed_date = today
            logger.info("Running morning swing review for all open positions")

            for symbol, scalper in self.scalpers.items():
                if not isinstance(scalper, SwingScalper):
                    continue
                scalper.reset_morning_review_flag()
                state = self.bar_manager.states.get(symbol)
                if state:
                    await scalper.morning_review(state)

    async def _session_guard_loop(self) -> None:
        tz = pytz.timezone(self.config.session.timezone)
        while self._running:
            await asyncio.sleep(30)
            now = datetime.now(tz)
            today = now.date().isoformat()

            if self._is_swing:
                # Swing mode: close of regular session is 16:00 — just record P&L, DON'T flatten
                end_h, end_m = map(int, self.config.swing.session_end_time.split(":"))
            else:
                end_h, end_m = map(int, self.config.session.end_time.split(":"))

            end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)

            if now > end and today != self._session_closed_date:
                self._session_closed_date = today
                if self._is_swing:
                    logger.info("Market closed — recording daily P&L (swing positions carried overnight)")
                else:
                    logger.info("Session ended — flattening positions")
                    self.positions.flatten_all()

                equity = self.executor.get_equity()
                trade_stats = self.journal.compute_trade_stats_for_date(today)
                starting = self.risk.state.starting_equity
                self.journal.upsert_daily_pnl(
                    today,
                    starting,
                    equity,
                    trade_stats["trade_count"],
                    trade_stats["win_count"],
                    net_pnl=trade_stats["net_pnl"],
                )
                summary_ctx = {
                    "date": today,
                    "equity": equity,
                    "trades": trade_stats["trade_count"],
                    "wins": trade_stats["win_count"],
                }
                summary = await self.llm.openclaw.daily_summary(summary_ctx)
                await self.llm.alert("Daily P&L Summary", summary)

            if self.risk.is_killed():
                logger.warning("Kill switch active — flattening and stopping")
                self.positions.flatten_all()
                await self.llm.alert("Kill Switch", self.risk.state.kill_reason)
                asyncio.create_task(self.stop())

    async def stop(self) -> None:
        """Stop streams and background tasks so the process can exit."""
        if self._stopping:
            return
        self._stopping = True
        logger.info("Shutting down agent...")
        self._running = False

        if self._stream:
            self._stream.stop()
        if self._trade_stream:
            self._trade_stream.stop()

        current = asyncio.current_task()
        tasks = [
            t
            for t in (
                self._watchlist_task,
                self._session_task,
                self._stream_task,
                self._trade_stream_task,
                self._morning_review_task,
                self._daily_screener_task,
            )
            if t and not t.done() and t is not current
        ]
        for task in tasks:
            task.cancel()

        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=5.0)
            if pending:
                logger.warning("%d task(s) still running after 5s — cancelling again", len(pending))
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        if self.risk.is_killed():
            # In swing mode, flatten all — kill switch overrides overnight holds
            self.positions.flatten_all()
            await self.llm.alert("Kill Switch", self.risk.state.kill_reason)
        elif not self._is_swing:
            # Scalper mode: always flatten on stop to avoid surprise overnight exposure
            self.positions.flatten_all()

        logger.info("Agent stopped")

    async def run(self) -> None:
        if not self.config.alpaca_api_key or not self.config.alpaca_secret_key:
            logger.error("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")
            return

        self._running = True
        equity = self.executor.get_equity()
        tz = pytz.timezone(self.config.session.timezone)
        today = datetime.now(tz).date()
        today_str = today.isoformat()
        starting_equity = self.journal.ensure_day_starting_equity(today_str, equity)
        self.risk.reset_session(equity, today, starting_equity=starting_equity)

        logger.info("Account equity: $%.2f", equity)
        llm_ok = await self.llm.check_health()
        primary = self.config.llm.resolved_primary()
        if primary == "openrouter" and self.config.llm.openrouter_configured():
            fallbacks = self.config.llm.openrouter_fallback_models
            logger.info(
                "LLM primary: OpenRouter (%s), reasoning=%s, fallbacks=%s, timeout=%ss",
                self.config.llm.openrouter_model,
                self.config.llm.openrouter_reasoning,
                ", ".join(fallbacks) if fallbacks else "none",
                self.config.llm.timeout_seconds,
            )
        elif primary == "google" and self.config.llm.google_api_key:
            logger.info(
                "LLM primary: Google (%s), timeout=%ss, rpm_limit=%s",
                self.config.llm.google_model,
                self.config.llm.timeout_seconds,
                self.config.llm.google_rpm_limit,
            )
        else:
            logger.info("Ollama health: %s (model=%s)", llm_ok, self.config.llm.ollama_model)

        await self._run_screener()
        await self._run_premarket_briefing()
        self._screener_ran_date = today_str

        self.active_symbols = self._merge_held_symbols(self.active_symbols)

        self.bar_manager.set_symbols(self.active_symbols)
        await self.bar_manager.warmup_async(self.active_symbols)

        self._stream = MarketDataStream(self.config, self.active_symbols, on_bar=self._on_bar)
        self._trade_stream = OrderUpdateStream(self.config, on_update=self._on_order_update)

        ScalperClass = SwingScalper if self._is_swing else SymbolScalper
        mode_label = "swing" if self._is_swing else "scalper"
        for symbol in self.active_symbols:
            self.scalpers[symbol] = ScalperClass(
                symbol,
                self.config,
                self.executor,
                self.risk,
                self.journal,
                self.llm,
                self._stream,
                avoided_symbols=self.avoided_symbols,
            )

        if self._is_swing:
            await hydrate_swing_positions(
                self.positions,
                self.scalpers,
                self.journal,
                self.bar_manager,
            )

        self._watchlist_task = asyncio.create_task(self._watchlist_loop())
        self._session_task = asyncio.create_task(self._session_guard_loop())
        self._stream_task = asyncio.create_task(self._stream.run())
        self._trade_stream_task = asyncio.create_task(self._trade_stream.run())
        self._daily_screener_task = asyncio.create_task(self._daily_screener_loop())

        if self._is_swing:
            self._morning_review_task = asyncio.create_task(self._morning_review_loop())
            logger.info(
                "Swing mode active — target %.1f%%, stop %.1f%%, hard_stop %.1f%%, max_hold %dd",
                self.config.swing.take_profit_pct,
                self.config.swing.stop_loss_pct,
                self.config.swing.hard_stop_pct,
                self.config.swing.max_hold_days,
            )

        await self.llm.alert(
            "Agent Started",
            f"Trading agent started [{mode_label}]. Symbols: {', '.join(self.active_symbols)}",
        )

        try:
            await asyncio.gather(self._stream_task, self._trade_stream_task)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()


def main() -> None:
    config = load_config()
    agent = TradingAgent(config)

    async def amain() -> None:
        loop = asyncio.get_running_loop()
        force_exit = False

        def on_signal() -> None:
            nonlocal force_exit
            if force_exit:
                logger.warning("Second interrupt — forcing exit")
                os._exit(130)
            force_exit = True
            logger.info("Shutdown signal received")
            asyncio.create_task(agent.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, on_signal)

        await agent.run()

    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
