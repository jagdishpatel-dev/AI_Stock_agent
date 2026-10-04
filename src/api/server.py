"""FastAPI dashboard API — journal, config, agent logs, and admin settings."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, load_config
from src.execution.orders import OrderExecutor
from src.execution.positions import PositionManager
from src.journal.logger import TradeJournal
from src.logging_sanitize import sanitize_log_message
from src.settings_store import (
    add_anchor,
    apply_patch,
    get_editable_snapshot,
    remove_anchor,
)

logger = logging.getLogger(__name__)

# No interactive docs or OpenAPI schema in production, and no CORS middleware:
# the dashboard is served from the same origin, so other sites can't call the API.
app = FastAPI(
    title="AI Trading Agent Dashboard",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_config = load_config()
_journal = TradeJournal(_config.journal_db_path)
_log_path = PROJECT_ROOT / "logs" / "agent.log"
_admin_key = os.getenv("ADMIN_API_KEY", "")

# Dashboard session: the frontend fetches /api/session to get a signed, HttpOnly
# cookie, and every other /api/* call requires it (or a valid X-Admin-Key).
# Without DASHBOARD_SESSION_SECRET a random secret is used per process; sessions
# then reset on restart and the frontend transparently fetches a new one.
_SESSION_COOKIE = "dash_session"
_SESSION_TTL_SECONDS = 12 * 60 * 60
_SESSION_EXEMPT_PATHS = {"/api/session", "/api/health"}
_session_secret = (os.getenv("DASHBOARD_SESSION_SECRET") or secrets.token_hex(32)).encode()


def _sign_session(expires: int) -> str:
    sig = hmac.new(_session_secret, str(expires).encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{sig}"


def _valid_session(token: str | None) -> bool:
    if not token:
        return False
    expires, _, sig = token.partition(".")
    if not expires.isdigit() or int(expires) < time.time():
        return False
    return hmac.compare_digest(sig, _sign_session(int(expires)).partition(".")[2])


def _valid_admin_key(key: str | None) -> bool:
    return bool(_admin_key) and key is not None and hmac.compare_digest(key, _admin_key)


@app.middleware("http")
async def _require_dashboard_session(request: Request, call_next):
    path = request.url.path
    if (
        path.startswith("/api/")
        and path not in _SESSION_EXEMPT_PATHS
        and not _valid_session(request.cookies.get(_SESSION_COOKIE))
        and not _valid_admin_key(request.headers.get("X-Admin-Key"))
    ):
        return JSONResponse({"detail": "Dashboard session required"}, status_code=401)
    return await call_next(request)


def _reload_runtime_config() -> None:
    global _config, _journal
    _config = load_config()
    _journal = TradeJournal(_config.journal_db_path)


def _require_admin(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")) -> None:
    if not _admin_key:
        raise HTTPException(
            status_code=503,
            detail="Admin API disabled — set ADMIN_API_KEY in .env on the server",
        )
    if not _valid_admin_key(x_admin_key):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")


class SettingsPatch(BaseModel):
    updates: dict[str, Any] = Field(default_factory=dict)


class AnchorRequest(BaseModel):
    symbol: str

# Recompute historical daily_pnl from trades on dashboard startup (idempotent).
try:
    _backfilled = _journal.backfill_daily_pnl()
    if _backfilled:
        logger.info("Backfilled %d daily_pnl row(s) on startup", _backfilled)
except Exception:
    logger.warning("daily_pnl backfill skipped", exc_info=True)


def _today_et() -> str:
    tz = pytz.timezone(_config.session.timezone)
    return datetime.now(tz).date().isoformat()


def _market_status() -> str:
    tz = pytz.timezone(_config.session.timezone)
    now = datetime.now(tz)
    if now.weekday() >= 5:
        return "weekend"
    start_h, start_m = map(int, _config.session.start_time.split(":"))
    if _config.strategy.mode == "swing":
        end_time = _config.swing.session_end_time
    else:
        end_time = _config.session.end_time
    end_h, end_m = map(int, end_time.split(":"))
    start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if now < start:
        return "pre_market"
    if now <= end:
        return "open"
    return "closed"


def _display_llm() -> str:
    provider = _config.llm.resolved_primary()
    if provider == "openrouter" and _config.llm.openrouter_configured():
        return f"openrouter / {_config.llm.openrouter_model}"
    if provider == "google" and _config.llm.google_api_key:
        return f"google / {_config.llm.google_model}"
    if provider == "ollama":
        return f"ollama / {_config.llm.ollama_model}"
    return f"{provider} / {_config.llm.openclaw_model}"


def _effective_session_end() -> str:
    if _config.strategy.mode == "swing":
        return _config.swing.session_end_time
    return _config.session.end_time


def _sanitize_config() -> dict[str, Any]:
    return {
        "symbols": _config.symbols,
        "strategy": _config.strategy.__dict__,
        "swing": _config.swing.__dict__,
        "session": _config.session.__dict__,
        "effective_session_end": _effective_session_end(),
        "display_llm": _display_llm(),
        "timezone": _config.session.timezone,
        "timezone_label": "CST",
        "risk": _config.risk.__dict__,
        "execution": _config.execution.__dict__,
        "briefing": _config.briefing.__dict__,
        "journal_context": _config.journal_context.__dict__,
        "screener": _config.screener.__dict__,
        "ai_exit": _config.ai_exit.__dict__,
        "llm": {
            "enabled": _config.llm.enabled,
            "primary_provider": _config.llm.resolved_primary(),
            "display_llm": _display_llm(),
            "confidence_threshold": _config.llm.confidence_threshold,
            "timeout_seconds": _config.llm.timeout_seconds,
            "watchlist_interval_minutes": _config.llm.watchlist_interval_minutes,
            "google_model": _config.llm.google_model,
            "google_thinking_level": _config.llm.google_thinking_level,
            "google_rpm_limit": _config.llm.google_rpm_limit,
            "has_google_key": bool(_config.llm.google_api_key),
            "ollama_model": _config.llm.ollama_model,
            "openclaw_model": _config.llm.openclaw_model,
            "alert_channel": _config.llm.alert_channel,
        },
        "journal_db_path": _config.journal_db_path,
        "alpaca_base_url": _config.alpaca_base_url,
        "alpaca_data_feed": _config.alpaca_data_feed,
        "has_alpaca_keys": bool(_config.alpaca_api_key),
        "has_finnhub_key": bool(_config.finnhub_api_key),
    }


def _fetch_account() -> dict[str, Any] | None:
    if not _config.alpaca_api_key or not _config.alpaca_secret_key:
        return None
    try:
        account = OrderExecutor(_config).get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(getattr(account, "portfolio_value", account.equity)),
            "last_equity": float(getattr(account, "last_equity", account.equity)),
        }
    except Exception:
        logger.warning("Failed to fetch Alpaca account", exc_info=True)
        return None


def _fetch_positions() -> list[dict[str, Any]]:
    if not _config.alpaca_api_key or not _config.alpaca_secret_key:
        return []
    try:
        raw = PositionManager(_config).list_positions()
        result: list[dict[str, Any]] = []
        for pos in raw:
            result.append(
                {
                    "symbol": str(pos.symbol),
                    "qty": float(pos.qty),
                    "side": "long" if float(pos.qty) > 0 else "short",
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc) * 100,
                }
            )
        return result
    except Exception:
        logger.warning("Failed to fetch Alpaca positions", exc_info=True)
        return []


def _previous_balance() -> float | None:
    rows = _journal.list_daily_pnl(limit=2)
    if not rows:
        return None
    today = _today_et()
    for row in rows:
        if row["date"] != today and row.get("ending_equity") is not None:
            return float(row["ending_equity"])
    if rows[0].get("starting_equity") is not None:
        return float(rows[0]["starting_equity"])
    return None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "today": _today_et(), "market_status": _market_status()}


@app.get("/api/session")
def session(request: Request, response: Response) -> dict[str, Any]:
    expires = int(time.time()) + _SESSION_TTL_SECONDS
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    response.set_cookie(
        _SESSION_COOKIE,
        _sign_session(expires),
        max_age=_SESSION_TTL_SECONDS,
        path="/api",
        httponly=True,
        samesite="strict",
        secure=proto == "https",
    )
    return {"ok": True, "expires": expires}


@app.get("/api/account")
def account() -> dict[str, Any]:
    data = _fetch_account()
    prev = _previous_balance()
    if data is None:
        return {"available": False, "previous_balance": prev}
    last_eq = data.get("last_equity")
    previous = prev if prev is not None else last_eq
    change = data["equity"] - previous if previous is not None else None
    change_pct = (change / previous * 100) if change is not None and previous else None
    return {
        "available": True,
        **data,
        "previous_balance": previous,
        "change": round(change, 2) if change is not None else None,
        "change_pct": round(change_pct, 4) if change_pct is not None else None,
    }


@app.get("/api/positions")
def positions() -> dict[str, Any]:
    items = _fetch_positions()
    account = _fetch_account()
    equity = account["equity"] if account else None
    for p in items:
        if equity and equity > 0:
            p["portfolio_pct"] = round(abs(p["market_value"]) / equity * 100, 2)
        else:
            p["portfolio_pct"] = None
    return {"positions": items, "count": len(items)}


@app.get("/api/portfolio")
def portfolio() -> dict[str, Any]:
    account = _fetch_account()
    positions_data = _fetch_positions()
    daily = _journal.list_daily_pnl(limit=90)
    lifetime = _journal.get_lifetime_stats()
    return {
        "account": account,
        "positions": positions_data,
        "daily_pnl": daily,
        "lifetime_stats": lifetime,
        "market_status": _market_status(),
        "today": _today_et(),
    }


@app.get("/api/overview")
def overview(date: str | None = None) -> dict[str, Any]:
    d = date or _today_et()
    summary = _journal.get_daily_summary(d)
    stats = _journal.get_stats(d)
    watchlist = _journal.get_daily_watchlist(d)
    account = _fetch_account()
    positions_data = _fetch_positions()
    lifetime = _journal.get_lifetime_stats()

    # Prefer trade-based P&L when daily_pnl row has zero but trades exist
    display_pnl = stats.get("total_pnl", 0)
    if summary and summary.get("net_pnl") is not None:
        if summary["net_pnl"] != 0 or stats.get("trade_count", 0) == 0:
            display_pnl = summary["net_pnl"]
        else:
            display_pnl = stats["total_pnl"]

    return {
        "date": d,
        "summary": summary,
        "stats": stats,
        "display_pnl": display_pnl,
        "watchlist": watchlist,
        "config": _sanitize_config(),
        "account": account,
        "positions": positions_data,
        "lifetime_stats": lifetime,
        "market_status": _market_status(),
        "daily_pnl_history": _journal.list_daily_pnl(limit=30),
        "last_trade_date": _journal.get_last_trade_date(),
    }


@app.get("/api/trades")
def trades(
    date: str | None = None,
    symbol: str | None = None,
    limit: int = Query(200, le=500),
    offset: int = 0,
) -> list[dict[str, Any]]:
    return _journal.list_trades(date=date, symbol=symbol, limit=limit, offset=offset)


@app.get("/api/round-trips")
def round_trips(
    date: str | None = None,
    limit: int = Query(100, le=300),
) -> list[dict[str, Any]]:
    return _journal.get_round_trips(date=date, limit=limit)


@app.get("/api/signals")
def signals(
    date: str | None = None,
    symbol: str | None = None,
    signal_type: str | None = None,
    limit: int = Query(200, le=500),
    offset: int = 0,
) -> list[dict[str, Any]]:
    return _journal.list_signals(
        date=date,
        symbol=symbol,
        signal_type=signal_type,
        limit=limit,
        offset=offset,
    )


@app.get("/api/events")
def events(
    date: str | None = None,
    event_type: str | None = None,
    limit: int = Query(200, le=500),
    offset: int = 0,
    _: None = Depends(_require_admin),
) -> list[dict[str, Any]]:
    return _journal.list_events(
        date=date,
        event_type=event_type,
        limit=limit,
        offset=offset,
    )


@app.get("/api/daily-pnl")
def daily_pnl(limit: int = Query(90, le=365)) -> list[dict[str, Any]]:
    return _journal.list_daily_pnl(limit=limit)


@app.get("/api/watchlist")
def watchlist(date: str | None = None) -> dict[str, Any]:
    d = date or _today_et()
    return {"date": d, "entries": _journal.get_daily_watchlist(d)}


@app.get("/api/watchlist/dates")
def watchlist_dates(limit: int = 30) -> list[str]:
    return _journal.list_watchlist_dates(limit=limit)


@app.get("/api/config")
def config(_: None = Depends(_require_admin)) -> dict[str, Any]:
    return _sanitize_config()


@app.get("/api/logs")
def logs(
    lines: int = Query(200, le=1000),
    _: None = Depends(_require_admin),
) -> dict[str, Any]:
    if not _log_path.exists():
        return {"path": str(_log_path), "lines": [], "exists": False}
    with open(_log_path) as f:
        tail = deque(f, maxlen=lines)
    return {
        "path": str(_log_path),
        "exists": True,
        "lines": [sanitize_log_message(line.rstrip("\n")) for line in tail],
    }


@app.get("/api/admin/settings")
def admin_get_settings(_: None = Depends(_require_admin)) -> dict[str, Any]:
    snapshot = get_editable_snapshot()
    snapshot["admin_enabled"] = bool(_admin_key)
    snapshot["note"] = (
        "Dashboard reloads config immediately. Restart trading-agent service "
        "for the agent to pick up changes."
    )
    return snapshot


@app.put("/api/admin/settings")
def admin_update_settings(
    body: SettingsPatch,
    _: None = Depends(_require_admin),
) -> dict[str, Any]:
    try:
        result = apply_patch(body.updates)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    _reload_runtime_config()
    return {"ok": True, **result, "settings": get_editable_snapshot()}


@app.post("/api/admin/anchors")
def admin_add_anchor(
    body: AnchorRequest,
    _: None = Depends(_require_admin),
) -> dict[str, Any]:
    try:
        anchors = add_anchor(body.symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    _reload_runtime_config()
    return {"ok": True, "anchor_symbols": anchors}


@app.delete("/api/admin/anchors/{symbol}")
def admin_remove_anchor(
    symbol: str,
    _: None = Depends(_require_admin),
) -> dict[str, Any]:
    try:
        anchors = remove_anchor(symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    _reload_runtime_config()
    return {"ok": True, "anchor_symbols": anchors}


_frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")


def main() -> None:
    import uvicorn

    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "8000"))
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
