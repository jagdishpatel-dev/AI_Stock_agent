"""Multi-day swing position manager.

Key differences from SymbolScalper:
- Holds positions overnight (does NOT flatten at session close)
- Uses momentum breakout entry instead of oversold bounce
- AI reviews open positions every morning at 09:15 ET
- Wider stops (1.0% dynamic, 1.5% hard) vs scalper (0.12%)
- Targets 2-3% in days rather than 0.20% in minutes
- PDT-safe: always holds at least min_hold_hours before selling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum

from src.config import AppConfig
from src.data.stream import MarketDataStream
from src.execution.orders import OrderExecutor
from src.journal.logger import TradeJournal, TradeRecord
from src.llm.router import LLMRouter
from src.risk.manager import RiskManager
from src.risk.sizing import calculate_position_size
from src.strategy.entry_context import attach_daily_price_context
from src.strategy.indicators import IndicatorState
from src.strategy.swing_signals import SignalType, evaluate_swing_entry, evaluate_swing_exit

logger = logging.getLogger(__name__)


class SwingState(str, Enum):
    IDLE = "idle"
    BUY_SUBMITTED = "buy_submitted"
    IN_POSITION = "in_position"
    SELL_SUBMITTED = "sell_submitted"


class SwingScalper:
    """Per-symbol swing trading state machine (multi-day holds)."""

    def __init__(
        self,
        symbol: str,
        config: AppConfig,
        executor: OrderExecutor,
        risk: RiskManager,
        journal: TradeJournal,
        llm: LLMRouter,
        stream: MarketDataStream,
        avoided_symbols: set[str] | None = None,
    ) -> None:
        self.symbol = symbol
        self.config = config
        self.swing = config.swing
        self.executor = executor
        self.risk = risk
        self.journal = journal
        self.llm = llm
        self.stream = stream
        self.avoided_symbols = avoided_symbols or set()
        self._avoid_block_logged = False

        self.state = SwingState.IDLE
        self.entry_price: float = 0.0
        self.highest_since_entry: float = 0.0
        self.position_qty: float = 0.0
        self.buy_order_id: str | None = None
        self.sell_order_id: str | None = None
        self.pending_exit_reason: str = ""
        self.entry_time: datetime | None = None
        self.entry_date: date | None = None
        self.dynamic_stop_pct: float = config.swing.stop_loss_pct
        self.morning_reviewed_today: bool = False

    def _pnl_pct(self, price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return ((price - self.entry_price) / self.entry_price) * 100

    def _days_held(self) -> int:
        if self.entry_date is None:
            return 0
        return (date.today() - self.entry_date).days

    def _hours_held(self) -> float:
        if self.entry_time is None:
            return 0.0
        return (datetime.now(timezone.utc) - self.entry_time).total_seconds() / 3600

    def _build_review_context(self, indicator_state: IndicatorState, close: float) -> dict:
        rsi = indicator_state.rsi(14)
        vwap_dev = indicator_state.vwap_deviation_pct()
        avg_vol = indicator_state.avg_volume()
        latest = indicator_state.bars[-1] if indicator_state.bars else None
        volume_ratio = (latest.volume / avg_vol if avg_vol and avg_vol > 0 else None)
        ema_fast = indicator_state.ema(self.swing.entry_ema_fast)
        ema_slow = indicator_state.ema(self.swing.entry_ema_slow)
        gap_pct = indicator_state.gap_pct_from_prev_close()

        return {
            "symbol": self.symbol,
            "entry_price": round(self.entry_price, 2),
            "current_price": round(close, 2),
            "pnl_pct": round(self._pnl_pct(close), 4),
            "days_held": self._days_held(),
            "hours_held": round(self._hours_held(), 1),
            "highest_since_entry": round(self.highest_since_entry, 2),
            "rsi": round(rsi, 2) if rsi is not None else None,
            "vwap_deviation_pct": round(vwap_dev, 4) if vwap_dev is not None else None,
            "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
            "ema_fast": round(ema_fast, 3) if ema_fast else None,
            "ema_slow": round(ema_slow, 3) if ema_slow else None,
            "gap_pct_from_prev_close": round(gap_pct, 3) if gap_pct is not None else None,
            "_swing_limits": {
                "take_profit_pct": self.swing.take_profit_pct,
                "stop_loss_pct": self.swing.stop_loss_pct,
                "hard_stop_pct": self.swing.hard_stop_pct,
                "trailing_stop_pct": self.swing.trailing_stop_pct,
                "max_hold_days": self.swing.max_hold_days,
            },
        }

    async def on_bar(self, indicator_state: IndicatorState) -> None:
        quote = self.stream.get_quote(self.symbol)
        close = indicator_state.latest_close()
        if close is None:
            return

        if self.state == SwingState.IN_POSITION:
            await self._handle_in_position(close, indicator_state)
            return

        if self.state != SwingState.IDLE:
            return

        if self.symbol in self.avoided_symbols:
            if not self._avoid_block_logged:
                self._avoid_block_logged = True
                self.journal.log_signal(
                    self.symbol,
                    "entry_vetoed",
                    "premarket_avoid",
                    "reject",
                    1.0,
                    "Symbol on pre-market avoid list for today",
                )
                logger.info(
                    "%s entry blocked — on pre-market avoid list for today",
                    self.symbol,
                )
            return

        entry_signal = evaluate_swing_entry(
            self.symbol,
            indicator_state,
            quote,
            self.swing,
            self.config.execution,
        )

        if entry_signal.signal_type != SignalType.BUY:
            return

        can_trade, reason = self.risk.can_open_position(self.symbol)
        if not can_trade:
            logger.info("%s swing entry blocked by risk: %s", self.symbol, reason)
            return

        # LLM veto (same pattern as scalper)
        llm_action = "approve"
        llm_confidence = 1.0
        llm_reason = "llm_disabled"
        ctx = dict(entry_signal.context)

        jc = self.config.journal_context
        if jc.enabled:
            stats = self.journal.get_similar_setup_stats(
                self.symbol,
                ctx.get("rsi", 0),
                ctx.get("vwap_deviation_pct", 0),
                rsi_tolerance=jc.rsi_tolerance,
                lookback_days=jc.lookback_days,
            )
            ctx["historical_stats"] = stats
            if (
                stats["similar_trades"] >= jc.min_trades_for_veto
                and stats["win_rate"] < jc.min_win_rate
            ):
                self.journal.log_signal(
                    self.symbol,
                    "entry_vetoed",
                    entry_signal.reason,
                    "reject",
                    1.0,
                    f"historical win_rate {stats['win_rate']:.2f} below {jc.min_win_rate}",
                    rsi=ctx.get("rsi"),
                    vwap_dev=ctx.get("vwap_deviation_pct"),
                    volume_ratio=ctx.get("volume_ratio"),
                )
                logger.info(
                    "%s swing entry blocked by journal history: %d trades, win_rate=%.2f",
                    self.symbol,
                    stats["similar_trades"],
                    stats["win_rate"],
                )
                return

        # Outcome cards feed the veto prompt even when hard journal veto is off.
        ctx["outcome_cards"] = self.journal.get_outcome_cards(
            symbol=self.symbol,
            rsi=ctx.get("rsi"),
            vwap_dev=ctx.get("vwap_deviation_pct"),
            rsi_tolerance=jc.rsi_tolerance,
            vwap_tolerance=jc.outcome_cards_vwap_tolerance,
            lookback_days=jc.lookback_days,
            limit=jc.outcome_cards_limit,
        )

        attach_daily_price_context(ctx, self.symbol, close)

        if self.config.llm.enabled:
            decision, source = await self.llm.trade_veto(ctx, swing=True)
            llm_action = decision.action
            llm_confidence = decision.confidence
            llm_reason = f"[{source}] {decision.reason}"

            if decision.action == "reject":
                self.journal.log_signal(
                    self.symbol,
                    "entry_vetoed",
                    entry_signal.reason,
                    llm_action,
                    llm_confidence,
                    llm_reason,
                )
                logger.info("%s LLM vetoed swing entry: %s", self.symbol, llm_reason)
                return

        self.journal.log_signal(
            self.symbol,
            "entry",
            entry_signal.reason,
            llm_action,
            llm_confidence,
            llm_reason,
            rsi=ctx.get("rsi"),
            vwap_dev=ctx.get("vwap_deviation_pct"),
            volume_ratio=ctx.get("volume_ratio"),
        )
        await self._submit_buy(close)

    async def _handle_in_position(self, close: float, indicator_state: IndicatorState) -> None:
        self.highest_since_entry = max(self.highest_since_entry, close)

        # Hard floor — no AI override possible
        exit_signal = evaluate_swing_exit(
            self.entry_price,
            close,
            self.highest_since_entry,
            self.dynamic_stop_pct,
            self.swing,
            indicator_state.rsi(14),
        )

        if exit_signal.signal_type == SignalType.SELL:
            await self._exit(close, exit_signal.reason, market=True)
            return

        # Force exit if max hold exceeded
        if self._days_held() >= self.swing.max_hold_days:
            logger.info("%s max_hold_days=%d reached — exiting", self.symbol, self.swing.max_hold_days)
            await self._exit(close, "max_hold_days", market=True)

    async def morning_review(self, indicator_state: IndicatorState) -> None:
        """Called at morning_review_time (09:15 ET) if a position is open.

        Asks the LLM: hold / exit / trail. Applies the decision:
        - exit: submit market sell
        - trail: tighten dynamic_stop_pct to lock profit
        - hold: do nothing, let position continue
        """
        if self.state != SwingState.IN_POSITION:
            return

        if self.morning_reviewed_today:
            logger.debug("%s already reviewed today", self.symbol)
            return

        self.morning_reviewed_today = True
        close = indicator_state.latest_close()
        if close is None:
            logger.warning("%s morning review: no price data", self.symbol)
            return

        self.highest_since_entry = max(self.highest_since_entry, close)

        # Hard stop check before LLM (no point asking if already below floor)
        pnl_pct = self._pnl_pct(close)
        if pnl_pct <= -self.swing.hard_stop_pct:
            await self._exit(close, "hard_stop_morning", market=True)
            return

        if not self.config.llm.enabled:
            logger.info("%s morning review skipped — LLM disabled", self.symbol)
            return

        ctx = self._build_review_context(indicator_state, close)
        jc = self.config.journal_context
        ctx["outcome_cards"] = self.journal.get_outcome_cards(
            symbol=self.symbol,
            rsi=ctx.get("rsi"),
            vwap_dev=ctx.get("vwap_deviation_pct"),
            rsi_tolerance=jc.rsi_tolerance,
            vwap_tolerance=jc.outcome_cards_vwap_tolerance,
            lookback_days=jc.lookback_days,
            limit=jc.outcome_cards_limit,
        )
        decision, source = await self.llm.swing_review(ctx)

        self.journal.log_signal(
            self.symbol,
            "swing_review",
            f"morning_review day={self._days_held()}",
            decision.action,
            decision.confidence,
            f"[{source}] stop_pct={decision.new_stop_pct} — {decision.reason}",
            rsi=ctx.get("rsi"),
            vwap_dev=ctx.get("vwap_deviation_pct"),
            volume_ratio=ctx.get("volume_ratio"),
        )

        logger.info(
            "%s morning review [%s]: action=%s confidence=%.2f stop=%s — %s",
            self.symbol,
            source,
            decision.action,
            decision.confidence,
            decision.new_stop_pct,
            decision.reason,
        )

        if decision.action == "exit":
            await self._exit(close, "ai_swing_exit_morning", market=True)
        elif decision.action == "trail" and decision.new_stop_pct is not None:
            new_stop = max(0.1, min(decision.new_stop_pct, self.swing.stop_loss_pct))
            self.dynamic_stop_pct = new_stop
            logger.info("%s trailing stop tightened to %.2f%%", self.symbol, new_stop)

    def reset_morning_review_flag(self) -> None:
        """Call once at the start of each new trading day."""
        self.morning_reviewed_today = False

    def hydrate_from_alpaca(
        self,
        *,
        entry_price: float,
        qty: float,
        current_price: float,
        entry_time: datetime | None = None,
    ) -> None:
        """Restore in-memory state from an Alpaca-held position after restart."""
        if entry_time is not None and entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        self.state = SwingState.IN_POSITION
        self.entry_price = entry_price
        self.position_qty = qty
        self.highest_since_entry = max(entry_price, current_price)
        self.dynamic_stop_pct = self.swing.stop_loss_pct
        self.entry_time = entry_time
        self.entry_date = entry_time.astimezone(timezone.utc).date() if entry_time else None
        self.morning_reviewed_today = False
        self.risk.register_open(self.symbol)

        logger.info(
            "%s hydrated from Alpaca: qty=%s entry=%.2f current=%.2f pnl=%.2f%%",
            self.symbol,
            qty,
            entry_price,
            current_price,
            self._pnl_pct(current_price),
        )

    async def check_hydration_exit(
        self,
        current_price: float,
        indicator_state: IndicatorState | None = None,
    ) -> bool:
        """Evaluate stop rules immediately after hydration. Returns True if exit submitted."""
        if self.state != SwingState.IN_POSITION:
            return False

        self.highest_since_entry = max(self.highest_since_entry, current_price)
        rsi = indicator_state.rsi(14) if indicator_state is not None else None
        exit_signal = evaluate_swing_exit(
            self.entry_price,
            current_price,
            self.highest_since_entry,
            self.dynamic_stop_pct,
            self.swing,
            rsi,
        )

        if exit_signal.signal_type == SignalType.SELL:
            await self._exit(current_price, f"{exit_signal.reason}_hydration", market=True)
            return True

        if self._days_held() >= self.swing.max_hold_days:
            await self._exit(current_price, "max_hold_days_hydration", market=True)
            return True

        return False

    async def _exit(self, price: float, reason: str, market: bool = True) -> None:
        self.journal.log_signal(self.symbol, "exit", reason)
        await self._submit_sell(price, reason, market=market)

    async def _submit_buy(self, price: float) -> None:
        equity = self.executor.get_equity()
        self.risk.update_equity(equity)

        # Use swing-specific risk params for sizing
        from src.config import RiskConfig, StrategyConfig
        swing_risk = RiskConfig(
            max_risk_per_trade_pct=self.swing.max_risk_per_trade_pct,
            max_open_positions=self.swing.max_open_positions,
            max_equity_pct_per_position=self.config.risk.max_equity_pct_per_position,
        )
        swing_strategy = StrategyConfig(stop_loss_pct=self.swing.stop_loss_pct)
        qty = calculate_position_size(equity, price, swing_risk, swing_strategy)
        if qty < 1:
            logger.info("%s swing buy skipped: qty < 1 at price %.2f", self.symbol, price)
            return

        self.state = SwingState.BUY_SUBMITTED
        order = self.executor.submit_market_buy(self.symbol, qty)
        self.buy_order_id = str(order.id)
        self.position_qty = qty
        logger.info("%s swing buy submitted: %s shares @ ~%.2f", self.symbol, qty, price)

    async def _submit_sell(self, price: float, reason: str, market: bool = True) -> None:
        if self.position_qty <= 0:
            return
        self.pending_exit_reason = reason
        self.executor.cancel_open_orders(self.symbol)
        self.state = SwingState.SELL_SUBMITTED
        order = self.executor.submit_market_sell(self.symbol, self.position_qty)
        self.sell_order_id = str(order.id)
        logger.info("%s swing sell submitted: %s shares (%s)", self.symbol, self.position_qty, reason)

    async def on_order_update(self, event: str, order_data: dict) -> None:
        if event != "fill":
            return

        order_id = str(order_data.get("order_id", order_data.get("id", "")))
        side = str(order_data.get("side", "")).lower()
        filled_qty = float(order_data.get("filled_qty", order_data.get("qty", 0)) or 0)
        filled_price = float(order_data.get("filled_avg_price", order_data.get("price", 0)) or 0)

        if order_id == self.buy_order_id and side in ("buy", ""):
            self.entry_price = filled_price
            self.highest_since_entry = filled_price
            self.position_qty = filled_qty
            self.state = SwingState.IN_POSITION
            self.entry_time = datetime.now(timezone.utc)
            self.entry_date = date.today()
            self.dynamic_stop_pct = self.swing.stop_loss_pct
            self.morning_reviewed_today = True  # don't review on entry day
            self.risk.register_open(self.symbol)
            self.journal.log_trade(
                TradeRecord(
                    symbol=self.symbol,
                    side="buy",
                    qty=filled_qty,
                    price=filled_price,
                    order_id=order_id,
                    reason="swing_entry_fill",
                )
            )
            await self.llm.alert(
                "Swing Entry",
                f"BUY {self.symbol} x {filled_qty} @ ${filled_price:.2f} | target ~{self.swing.take_profit_pct}%",
            )
            return

        is_sell_fill = side == "sell" or (
            self.state in (SwingState.IN_POSITION, SwingState.SELL_SUBMITTED) and order_id != self.buy_order_id
        )
        if is_sell_fill and self.state in (SwingState.IN_POSITION, SwingState.SELL_SUBMITTED):
            pnl = (filled_price - self.entry_price) * filled_qty if self.entry_price else 0
            days = self._days_held()
            self.risk.record_realized_pnl(pnl)
            self.risk.register_close(self.symbol)
            exit_reason = self.pending_exit_reason or "exit_fill"
            self.journal.log_trade(
                TradeRecord(
                    symbol=self.symbol,
                    side="sell",
                    qty=filled_qty,
                    price=filled_price,
                    order_id=order_id,
                    pnl=pnl,
                    reason=exit_reason,
                )
            )
            await self.llm.alert(
                "Swing Exit",
                f"SELL {self.symbol} x {filled_qty} @ ${filled_price:.2f} | P&L: ${pnl:.2f} | held {days}d | reason: {exit_reason}",
            )
            self._reset()

    def _reset(self) -> None:
        self.state = SwingState.IDLE
        self.entry_price = 0.0
        self.highest_since_entry = 0.0
        self.position_qty = 0.0
        self.buy_order_id = None
        self.sell_order_id = None
        self.pending_exit_reason = ""
        self.entry_time = None
        self.entry_date = None
        self.dynamic_stop_pct = self.swing.stop_loss_pct
        self.morning_reviewed_today = False
