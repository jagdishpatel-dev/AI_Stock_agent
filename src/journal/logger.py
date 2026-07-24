"""SQLite trade journal for signals, fills, and P&L tracking."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass
class TradeRecord:
    symbol: str
    side: str
    qty: float
    price: float
    order_id: str
    pnl: float | None = None
    reason: str = ""


class TradeJournal:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    details TEXT,
                    llm_action TEXT,
                    llm_confidence REAL,
                    llm_reason TEXT,
                    rsi REAL,
                    vwap_dev REAL,
                    volume_ratio REAL
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    order_id TEXT,
                    pnl REAL,
                    reason TEXT
                );
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date TEXT PRIMARY KEY,
                    starting_equity REAL,
                    ending_equity REAL,
                    net_pnl REAL,
                    trade_count INTEGER,
                    win_count INTEGER
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT
                );
                CREATE TABLE IF NOT EXISTS daily_watchlist (
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    source TEXT,
                    rank INTEGER,
                    metrics_json TEXT,
                    reason TEXT,
                    PRIMARY KEY (date, symbol)
                );
                """
            )
            self._migrate_signals_columns(conn)

    def _migrate_signals_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
        for col, col_type in [("rsi", "REAL"), ("vwap_dev", "REAL"), ("volume_ratio", "REAL")]:
            if col not in existing:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {col_type}")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        details: str = "",
        llm_action: str | None = None,
        llm_confidence: float | None = None,
        llm_reason: str | None = None,
        rsi: float | None = None,
        vwap_dev: float | None = None,
        volume_ratio: float | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO signals (ts, symbol, signal_type, details, llm_action, llm_confidence,
                   llm_reason, rsi, vwap_dev, volume_ratio)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self._now(),
                    symbol,
                    signal_type,
                    details,
                    llm_action,
                    llm_confidence,
                    llm_reason,
                    rsi,
                    vwap_dev,
                    volume_ratio,
                ),
            )

    def log_trade(self, record: TradeRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO trades (ts, symbol, side, qty, price, order_id, pnl, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self._now(),
                    record.symbol,
                    record.side,
                    record.qty,
                    record.price,
                    record.order_id,
                    record.pnl,
                    record.reason,
                ),
            )

    def log_event(self, event_type: str, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (ts, event_type, message) VALUES (?, ?, ?)",
                (self._now(), event_type, message),
            )

    def ensure_day_starting_equity(self, date: str, equity: float) -> float:
        """Record starting equity once per day; return persisted value on restarts."""
        existing = self.get_daily_summary(date)
        if existing and existing.get("starting_equity") is not None:
            return float(existing["starting_equity"])
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO daily_pnl (date, starting_equity, ending_equity, net_pnl, trade_count, win_count)
                   VALUES (?, ?, ?, 0, 0, 0)
                   ON CONFLICT(date) DO NOTHING""",
                (date, equity, equity),
            )
        row = self.get_daily_summary(date)
        return float(row["starting_equity"]) if row else equity

    def compute_trade_stats_for_date(self, date: str) -> dict[str, Any]:
        """Aggregate realized P&L from sell fills for a calendar date (YYYY-MM-DD)."""
        trades = self.get_today_trades(date)
        sells = [t for t in trades if t["side"] == "sell" and t.get("pnl") is not None]
        wins = sum(1 for t in sells if t["pnl"] > 0)
        net_pnl = sum(t["pnl"] for t in sells)
        return {
            "net_pnl": round(net_pnl, 2),
            "trade_count": len(sells),
            "win_count": wins,
        }

    def upsert_daily_pnl(
        self,
        date: str,
        starting_equity: float,
        ending_equity: float,
        trade_count: int,
        win_count: int,
        net_pnl: float | None = None,
    ) -> None:
        if net_pnl is None:
            net_pnl = ending_equity - starting_equity
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO daily_pnl (date, starting_equity, ending_equity, net_pnl, trade_count, win_count)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(date) DO UPDATE SET
                     starting_equity=excluded.starting_equity,
                     ending_equity=excluded.ending_equity,
                     net_pnl=excluded.net_pnl,
                     trade_count=excluded.trade_count,
                     win_count=excluded.win_count""",
                (date, starting_equity, ending_equity, net_pnl, trade_count, win_count),
            )

    def backfill_daily_pnl(self) -> int:
        """Recompute net_pnl and trade counts from trades table for all daily_pnl rows."""
        with self._connect() as conn:
            dates = [r["date"] for r in conn.execute("SELECT date FROM daily_pnl ORDER BY date").fetchall()]
        updated = 0
        for d in dates:
            stats = self.compute_trade_stats_for_date(d)
            summary = self.get_daily_summary(d)
            if not summary:
                continue
            self.upsert_daily_pnl(
                d,
                float(summary["starting_equity"]),
                float(summary["ending_equity"]),
                stats["trade_count"],
                stats["win_count"],
                net_pnl=stats["net_pnl"],
            )
            updated += 1
        return updated

    def get_lifetime_stats(self) -> dict[str, Any]:
        """All-time realized P&L and win rate from sell trades."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pnl FROM trades WHERE side = 'sell' AND pnl IS NOT NULL"
            ).fetchall()
        pnls = [float(r["pnl"]) for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "total_pnl": round(sum(pnls), 2),
            "trade_count": len(pnls),
            "win_count": wins,
            "loss_count": len(pnls) - wins,
            "win_rate": round(wins / len(pnls), 3) if pnls else 0.0,
        }

    def get_last_trade_date(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT ts FROM trades ORDER BY ts DESC LIMIT 1").fetchone()
        if not row:
            return None
        return str(row["ts"])[:10]

    def get_today_trades(self, date_prefix: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE ts LIKE ? ORDER BY ts",
                (f"{date_prefix}%",),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_open_lot(self, symbol: str) -> dict[str, Any] | None:
        """Return metadata for the current open lot (most recent unmatched buy), if any."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE symbol = ? ORDER BY ts ASC",
                (symbol.upper(),),
            ).fetchall()

        if not rows:
            return None

        open_qty = 0.0
        lot_entry: dict[str, Any] | None = None
        for row in rows:
            qty = float(row["qty"])
            if str(row["side"]).lower() == "buy":
                if open_qty <= 0:
                    lot_entry = dict(row)
                open_qty += qty
            else:
                open_qty -= qty
                if open_qty <= 0:
                    lot_entry = None
                    open_qty = 0.0

        if open_qty > 0 and lot_entry is not None:
            lot_entry["open_qty"] = open_qty
            return lot_entry
        return None

    def get_daily_summary(self, date: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM daily_pnl WHERE date = ?", (date,)).fetchone()
        return dict(row) if row else None

    def get_similar_setup_stats(
        self,
        symbol: str,
        rsi: float,
        vwap_dev: float,
        rsi_tolerance: float = 5.0,
        vwap_tolerance: float = 0.1,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        with self._connect() as conn:
            signals = conn.execute(
                """SELECT id, ts, symbol FROM signals
                   WHERE symbol = ? AND signal_type = 'entry' AND ts >= ?
                   AND rsi IS NOT NULL AND vwap_dev IS NOT NULL
                   AND ABS(rsi - ?) < ? AND ABS(vwap_dev - ?) < ?""",
                (symbol, cutoff, rsi, rsi_tolerance, vwap_dev, vwap_tolerance),
            ).fetchall()

            pnls: list[float] = []
            for sig in signals:
                sell = conn.execute(
                    """SELECT pnl FROM trades
                       WHERE symbol = ? AND side = 'sell' AND pnl IS NOT NULL AND ts > ?
                       ORDER BY ts ASC LIMIT 1""",
                    (symbol, sig["ts"]),
                ).fetchone()
                if sell and sell["pnl"] is not None:
                    pnls.append(float(sell["pnl"]))

        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        total = len(pnls)
        return {
            "similar_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total, 3) if total else 0.0,
            "avg_pnl": round(sum(pnls) / total, 2) if total else 0.0,
        }

    def get_outcome_cards(
        self,
        *,
        symbol: str | None = None,
        rsi: float | None = None,
        vwap_dev: float | None = None,
        rsi_tolerance: float = 5.0,
        vwap_tolerance: float = 0.25,
        lookback_days: int = 30,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Build short outcome cards from recent closed round-trips for LLM prompts.

        Prefers similar RSI/VWAP setups, then same-symbol history, then any recent closes.
        Example summary: ``last 5 similar: 1W/4L, avg_pnl=$-50.12``.
        """
        if limit <= 0:
            return {"summary": "outcome cards disabled", "similar_count": 0, "cards": []}

        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        trips = [
            t
            for t in self.get_round_trips(limit=300)
            if (t.get("sell_ts") or "") >= cutoff and t.get("pnl") is not None
        ]

        def _similar(t: dict[str, Any]) -> bool:
            if rsi is None or vwap_dev is None:
                return False
            tr, tv = t.get("rsi"), t.get("vwap_dev")
            if tr is None or tv is None:
                return False
            return abs(float(tr) - float(rsi)) < rsi_tolerance and abs(
                float(tv) - float(vwap_dev)
            ) < vwap_tolerance

        similar = [t for t in trips if _similar(t)]
        same_symbol = [t for t in trips if symbol and t.get("symbol") == symbol]
        pools = (similar, same_symbol, trips)

        seen: set[str] = set()
        selected: list[dict[str, Any]] = []
        for pool in pools:
            for t in pool:
                key = str(t.get("sell_order_id") or f"{t.get('symbol')}-{t.get('sell_ts')}")
                if key in seen:
                    continue
                seen.add(key)
                selected.append(t)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break

        cards: list[dict[str, Any]] = []
        for t in selected:
            pnl = float(t["pnl"])
            cards.append(
                {
                    "symbol": t.get("symbol"),
                    "pnl": round(pnl, 2),
                    "result": "win" if pnl > 0 else "loss",
                    "exit_reason": t.get("exit_reason"),
                    "entry_reason": t.get("entry_reason"),
                    "rsi": t.get("rsi"),
                    "vwap_dev": t.get("vwap_dev"),
                    "volume_ratio": t.get("volume_ratio"),
                }
            )

        wins = sum(1 for c in cards if c["result"] == "win")
        losses = len(cards) - wins
        avg_pnl = sum(float(c["pnl"]) for c in cards) / len(cards) if cards else 0.0
        if cards:
            summary = f"last {len(cards)} similar: {wins}W/{losses}L, avg_pnl=${avg_pnl:.2f}"
        else:
            summary = "no recent closed trades"

        return {
            "summary": summary,
            "similar_count": len(similar),
            "same_symbol_count": len(same_symbol),
            "cards": cards,
        }

    def save_daily_watchlist(
        self,
        date: str,
        entries: list[dict],
    ) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM daily_watchlist WHERE date = ?", (date,))
            for entry in entries:
                conn.execute(
                    """INSERT INTO daily_watchlist (date, symbol, source, rank, metrics_json, reason)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        date,
                        entry["symbol"],
                        entry.get("source", ""),
                        entry.get("rank", 0),
                        json.dumps(entry.get("metrics", {})),
                        entry.get("reason", ""),
                    ),
                )

    def get_daily_watchlist(self, date: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_watchlist WHERE date = ? ORDER BY rank",
                (date,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["metrics"] = json.loads(d.get("metrics_json") or "{}")
            except json.JSONDecodeError:
                d["metrics"] = {}
            result.append(d)
        return result

    def list_trades(
        self,
        *,
        date: str | None = None,
        symbol: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if date:
            clauses.append("ts LIKE ?")
            params.append(f"{date}%")
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM trades {where} ORDER BY ts DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def list_signals(
        self,
        *,
        date: str | None = None,
        symbol: str | None = None,
        signal_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if date:
            clauses.append("ts LIKE ?")
            params.append(f"{date}%")
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if signal_type:
            clauses.append("signal_type = ?")
            params.append(signal_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM signals {where} ORDER BY ts DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def list_events(
        self,
        *,
        date: str | None = None,
        event_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if date:
            clauses.append("ts LIKE ?")
            params.append(f"{date}%")
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM events {where} ORDER BY ts DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def list_daily_pnl(self, limit: int = 90) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_pnl ORDER BY date DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_watchlist_dates(self, limit: int = 30) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT date FROM daily_watchlist ORDER BY date DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["date"] for r in rows]

    def get_round_trips(self, *, date: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Pair buy/sell trades with entry/exit signals and LLM context."""
        trades = self.list_trades(date=date, limit=500)
        trades.sort(key=lambda t: t["ts"])
        open_buys: dict[str, dict[str, Any]] = {}
        rounds: list[dict[str, Any]] = []

        for trade in trades:
            sym = trade["symbol"]
            if trade["side"] == "buy":
                entry_signal = self._nearest_signal(sym, trade["ts"], ("entry",))
                open_buys[sym] = {
                    "symbol": sym,
                    "buy_ts": trade["ts"],
                    "buy_price": trade["price"],
                    "buy_qty": trade["qty"],
                    "buy_order_id": trade.get("order_id"),
                    "entry_reason": entry_signal.get("details") if entry_signal else None,
                    "llm_action": entry_signal.get("llm_action") if entry_signal else None,
                    "llm_confidence": entry_signal.get("llm_confidence") if entry_signal else None,
                    "llm_reason": entry_signal.get("llm_reason") if entry_signal else None,
                    "rsi": entry_signal.get("rsi") if entry_signal else None,
                    "vwap_dev": entry_signal.get("vwap_dev") if entry_signal else None,
                    "volume_ratio": entry_signal.get("volume_ratio") if entry_signal else None,
                }
            elif trade["side"] == "sell" and sym in open_buys:
                buy = open_buys.pop(sym)
                exit_signal = self._nearest_signal(sym, trade["ts"], ("exit",))
                exit_reason = trade.get("reason") or "unknown"
                if exit_signal and exit_reason in ("exit_fill", "unknown"):
                    exit_reason = exit_signal.get("details") or exit_reason
                rounds.append(
                    {
                        **buy,
                        "sell_ts": trade["ts"],
                        "sell_price": trade["price"],
                        "sell_qty": trade["qty"],
                        "sell_order_id": trade.get("order_id"),
                        "exit_reason": exit_reason,
                        "pnl": trade.get("pnl"),
                    }
                )

        rounds.sort(key=lambda r: r.get("sell_ts") or r.get("buy_ts"), reverse=True)
        return rounds[:limit]

    def _nearest_signal(
        self,
        symbol: str,
        trade_ts: str,
        signal_types: tuple[str, ...],
    ) -> dict[str, Any] | None:
        placeholders = ",".join("?" for _ in signal_types)
        with self._connect() as conn:
            row = conn.execute(
                f"""SELECT * FROM signals
                    WHERE symbol = ? AND signal_type IN ({placeholders}) AND ts <= ?
                    ORDER BY ts DESC LIMIT 1""",
                (symbol, *signal_types, trade_ts),
            ).fetchone()
        return dict(row) if row else None

    def get_stats(self, date: str | None = None) -> dict[str, Any]:
        trades = self.list_trades(date=date, limit=1000)
        sells = [t for t in trades if t["side"] == "sell" and t.get("pnl") is not None]
        wins = sum(1 for t in sells if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in sells)
        signals = self.list_signals(date=date, limit=1000)
        vetoes = [s for s in signals if s["signal_type"] == "entry_vetoed"]
        entries = [s for s in signals if s["signal_type"] == "entry"]
        return {
            "trade_count": len(sells),
            "win_count": wins,
            "loss_count": len(sells) - wins,
            "win_rate": round(wins / len(sells), 3) if sells else 0.0,
            "total_pnl": round(total_pnl, 2),
            "entry_signals": len(entries),
            "llm_vetoes": len(vetoes),
            "exit_signals": sum(1 for s in signals if s["signal_type"] == "exit"),
        }
