"""Read/write config/settings.yaml for the admin portal."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT

SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Keys the admin UI may update (nested dict paths as tuples).
EDITABLE_PATHS: list[tuple[str, ...]] = [
    ("strategy", "mode"),
    ("screener", "enabled"),
    ("screener", "mode"),
    ("screener", "dynamic_slots"),
    ("screener", "candidate_pool_size"),
    ("screener", "run_time"),
    ("screener", "anchor_symbols"),
    ("swing", "take_profit_pct"),
    ("swing", "stop_loss_pct"),
    ("swing", "hard_stop_pct"),
    ("swing", "max_hold_days"),
    ("swing", "max_open_positions"),
    ("swing", "max_risk_per_trade_pct"),
    ("risk", "max_open_positions"),
    ("risk", "max_equity_pct_per_position"),
    ("risk", "max_risk_per_trade_pct"),
    ("risk", "daily_max_loss_pct"),
    ("llm", "enabled"),
    ("llm", "primary_provider"),
    ("llm", "confidence_threshold"),
    ("llm", "watchlist_interval_minutes"),
    ("briefing", "enabled"),
]


def load_raw() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    with open(SETTINGS_PATH) as f:
        return yaml.safe_load(f) or {}


def save_raw(data: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return copy.deepcopy(cur)


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = data
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def get_editable_snapshot() -> dict[str, Any]:
    raw = load_raw()
    snapshot: dict[str, Any] = {}
    for path in EDITABLE_PATHS:
        key = ".".join(path)
        snapshot[key] = _get_nested(raw, path)
    snapshot["settings_path"] = str(SETTINGS_PATH)
    return snapshot


def apply_patch(updates: dict[str, Any]) -> dict[str, Any]:
    """Apply flat dotted-key updates. Returns applied keys."""
    raw = load_raw()
    applied: list[str] = []
    for dotted, value in updates.items():
        if dotted in ("settings_path",):
            continue
        path = tuple(dotted.split("."))
        if path not in EDITABLE_PATHS:
            raise ValueError(f"Setting not editable: {dotted}")
        _set_nested(raw, path, value)
        applied.append(dotted)
    save_raw(raw)
    return {"applied": applied}


def normalize_symbol(symbol: str) -> str:
    sym = symbol.strip().upper()
    if not sym or not sym.isalpha() or len(sym) > 6:
        raise ValueError(f"Invalid symbol: {symbol}")
    return sym


def add_anchor(symbol: str) -> list[str]:
    raw = load_raw()
    anchors = list(_get_nested(raw, ("screener", "anchor_symbols")) or [])
    sym = normalize_symbol(symbol)
    if sym not in anchors:
        anchors.append(sym)
    _set_nested(raw, ("screener", "anchor_symbols"), anchors)
    save_raw(raw)
    return anchors


def remove_anchor(symbol: str) -> list[str]:
    raw = load_raw()
    anchors = list(_get_nested(raw, ("screener", "anchor_symbols")) or [])
    sym = normalize_symbol(symbol)
    anchors = [a for a in anchors if a.upper() != sym]
    _set_nested(raw, ("screener", "anchor_symbols"), anchors)
    save_raw(raw)
    return anchors
