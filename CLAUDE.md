# CLAUDE.md

Guide for AI coding agents (including the nightly maintenance + R&D agent) working in this repo. Read this first. For the full architecture, see `docs/SYSTEM_REPORT.md`; for setup and the API, see `README.md`.

## What this is

An automated US equities trading agent on **Alpaca paper trading**, plus a React dashboard. Core design rule: **fast Python rules make every trade decision on 1-minute bars; LLMs only review, rank or veto — they never initiate trades.**

Active strategy mode is set in `config/settings.yaml` → `strategy.mode` (currently `swing`; `scalper` also exists).

## Map of the code

| Path | What it does |
|------|--------------|
| `src/main.py` | `TradingAgent` orchestrator: screener → pre-market briefing → bar stream → signals → ReAct watchlist ranking → risk → orders; morning review and session guard loops |
| `src/strategy/` | Rule-based signals. `swing_scalper.py` / `swing_signals.py` (swing mode), `scalper.py` / `signals.py` (scalper mode), `indicators.py`, `entry_context.py` / `exit_context.py` (context built for LLM review) |
| `src/llm/router.py` | `LLMRouter`: provider fallback chain (OpenRouter → Google → Ollama → OpenClaw) and the LLM tasks: `trade_veto`, `rank_watchlist`, `briefing_decision`, `screener_rank`, `exit_advisor`, `swing_review` |
| `src/llm/react.py` | ReAct watchlist ranking — read-only tools, max 4 tool calls + 1 answer per run, 5 runs/day; bottom N symbols gated from new entries |
| `src/llm/prompts.py` | Prompts for all LLM tasks |
| `src/journal/logger.py` | SQLite `TradeJournal` (`data/trades.db`): signals, trades, events, daily P&L, round trips. Also the current learning-from-history pieces: `get_similar_setup_stats` and `get_outcome_cards` (fed back into veto decisions via `journal_context` in settings) |
| `src/risk/` | Kill switch, daily loss limit, max positions, PDT rules, position sizing |
| `src/execution/` | Bracket orders, position tracking, state hydration on restart |
| `src/data/` | Alpaca bar/trade streams, Yahoo Finance research data, news parsing |
| `src/screener/` | Daily watchlist builder (fetch → filter → enrich → rank) |
| `src/api/server.py` | FastAPI dashboard backend (session cookie or `X-Admin-Key` on every `/api/*` except `/api/health` and `/api/session`) |
| `frontend/` | React 19 + TypeScript + Vite dashboard |
| `scripts/` | Reports, `backtest.py`, deploy and status scripts |
| `tests/` | pytest suite |

## Running checks

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q                                   # all tests run offline, no API keys needed
ruff check src tests scripts --select E9,F63,F7,F82   # syntax / undefined-name errors (what CI enforces)
cd frontend && npm ci && npm run build                # TypeScript + Vite build
```

- Tests must not need real Alpaca, LLM or network access. Mock them (see existing tests for patterns: `monkeypatch`, fake routers, `tmp_path` SQLite DBs).
- `npm run build` rewrites `frontend/tsconfig.tsbuildinfo`; don't commit that change unless you meant to.
- `scripts/backtest.py` currently replays the **scalper** signals (`src/strategy/signals.py`), not the swing strategy that is live, and needs Alpaca keys.

## ⚠️ Deployment: merging = shipping

`.github/workflows/deploy.yml` runs `scripts/deploy.sh` on a **self-hosted runner on the home server on every push to `main`**. Merging a PR deploys it and restarts the dashboard (the agent restart is skipped during market hours, 8:30–15:00 CT). So:

- Never push to `main` directly. Always work on a branch and open a PR.
- In PR descriptions, say whether the change affects live behaviour and suggest merging outside market hours if it does.

## Rules for AI agents

**Off-limits — open an issue with your analysis instead of changing code:**
- Trading strategy and signal logic, entry/exit thresholds
- Risk limits, position sizing, kill switch, PDT logic
- Order placement logic
- Prompts, models or providers that drive live trade decisions
- Wiring any self-improvement loop into live decisions
- Values in `config/settings.yaml` that change trading behaviour

**Never touch:** `.env` / secrets / credentials, `.github/workflows/deploy.yml`, `.github/workflows/check-status.yml`, `scripts/deploy.sh`, `systemd/`.

**Workflows:** this is a public repo with a self-hosted runner. Any new workflow that runs on pull requests **must use GitHub-hosted runners (`ubuntu-latest`)**, never `self-hosted`, so PR code never runs on the home server.

**Good targets for agent PRs:** bug fixes with tests, crash-proofing (timeouts, retries, error handling around Alpaca / webhook / LLM calls), dependency fixes, tests, decision/outcome logging, evaluation and replay tooling, reports.

## Conventions

- Python 3.11+, type hints, `from __future__ import annotations`, asyncio for anything I/O.
- Log through the journal (`log_signal`, `log_trade`, `log_event`) so the dashboard can show it; scrub secrets with `src/logging_sanitize.py`.
- Keep PRs small and focused; include how you verified the change.

## Current goals

- Win rate has dropped a lot from earlier highs — measuring and improving decision quality is the top priority.
- Building a self-improving component on top of the journal (outcome cards / similar-setup stats are the starting point).
- Needed first: a reliable way to evaluate a change offline (replay of the live swing strategy against past data and logged decisions) before it touches paper trading.
