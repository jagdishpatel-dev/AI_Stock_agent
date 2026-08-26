"""Load settings from YAML and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class StrategyConfig:
    mode: str = "scalper"  # scalper | swing
    rsi_period: int = 14
    rsi_oversold: float = 35
    rsi_overbought: float = 65
    volume_spike_ratio: float = 1.5
    vwap_deviation_pct: float = 0.15
    take_profit_pct: float = 0.20
    stop_loss_pct: float = 0.12
    trailing_stop_pct: float = 0.10


@dataclass
class SwingConfig:
    """Parameters for the multi-day swing trading mode."""
    # Targets and stops (all in %)
    take_profit_pct: float = 2.5       # AI review decides whether to hold past this
    stop_loss_pct: float = 1.0         # Regular stop
    hard_stop_pct: float = 1.5         # Absolute floor, never overridden by AI
    trailing_stop_pct: float = 0.5     # Trail from recent high (activates after profit_lock_pct)
    profit_lock_pct: float = 0.8       # Start trailing once up this much
    # Hold duration
    max_hold_days: int = 5             # Force exit after this many calendar days
    min_hold_hours: float = 4.0        # Minimum hold before AI exit allowed (intraday filter)
    # Entry signals
    entry_min_gap_pct: float = 0.3     # Min gap from previous close to confirm momentum
    entry_max_gap_pct: float = 8.0     # Avoid parabolic gaps (gap-and-crap risk)
    entry_min_volume_ratio: float = 1.5
    entry_ema_fast: int = 10
    entry_ema_slow: int = 20
    entry_max_rsi: float = 72.0        # Don't chase overbought
    # Session behaviour
    flatten_on_close: bool = False     # True = scalper style; False = hold overnight
    morning_review_time: str = "08:15"  # AI reviews open positions before regular open (CST)
    session_end_time: str = "15:00"     # Trade until market close (CST)
    # Sizing
    max_open_positions: int = 3
    max_risk_per_trade_pct: float = 1.5


@dataclass
class SessionConfig:
    timezone: str = "America/Chicago"
    start_time: str = "08:30"
    end_time: str = "10:30"


@dataclass
class RiskConfig:
    max_risk_per_trade_pct: float = 1.0
    daily_max_loss_pct: float = 2.0
    max_open_positions: int = 2
    max_equity_pct_per_position: float = 10.0
    pdt_equity_threshold: float = 25000
    max_day_trades: int = 3


@dataclass
class ExecutionConfig:
    max_spread_pct: float = 0.03
    use_bracket_orders: bool = True


@dataclass
class AIExitConfig:
    enabled: bool = True
    profit_trigger_pct: float = 0.40
    loss_trigger_pct: float = -0.06
    min_take_profit_pct: float = 0.20
    max_target_pct: float = 1.0
    max_hold_minutes: int = 20
    max_loss_hold_minutes: int = 8
    recheck_interval_bars: int = 5
    bar_context_count: int = 30
    hard_stop_loss_pct: float = 0.12


@dataclass
class BriefingConfig:
    enabled: bool = True
    time: str = "09:00"
    news_lookback_hours: int = 24
    news_limit: int = 10


@dataclass
class JournalContextConfig:
    enabled: bool = True
    rsi_tolerance: float = 5.0
    lookback_days: int = 30
    min_trades_for_veto: int = 5
    min_win_rate: float = 0.4
    outcome_cards_limit: int = 5
    outcome_cards_vwap_tolerance: float = 0.25


@dataclass
class ScreenerConfig:
    enabled: bool = True
    mode: str = "hybrid"
    anchor_symbols: list[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    dynamic_slots: int = 3
    run_time: str = "08:45"
    candidate_pool_size: int = 30
    min_price: float = 15.0
    max_price: float = 500.0
    min_volume: float = 1000000
    max_percent_change: float = 8.0
    exclude_sectors: list[str] = field(default_factory=list)
    finnhub_enabled: bool = True
    exclude_yesterday_losers: bool = True


@dataclass
class ResearchConfig:
    provider: str = "yahoo"  # yahoo | alpaca
    yahoo_enabled: bool = True
    finnhub_fallback: bool = True
    warmup_min_bars: int = 30


@dataclass
class LLMConfig:
    enabled: bool = True
    primary_provider: str = "auto"
    confidence_threshold: float = 0.7
    timeout_seconds: float = 3.0
    watchlist_interval_minutes: int = 10
    google_api_key: str = ""
    google_model: str = "gemma-4-31b-it"
    google_thinking_level: str = "MINIMAL"
    google_rpm_limit: int = 12
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "phi4-mini"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = ""
    openrouter_fallback_models: list[str] = field(default_factory=list)
    openrouter_reasoning: bool = False
    openclaw_gateway_url: str = "http://127.0.0.1:18789"
    openclaw_model: str = "gemma4:cloud"
    alert_channel: str = "telegram"
    alert_webhook_url: str = ""

    def openrouter_configured(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_model)

    def resolved_primary(self) -> str:
        """Return active LLM provider: openrouter, google, or ollama."""
        if self.primary_provider == "openrouter":
            if self.openrouter_configured():
                return "openrouter"
            return "google" if self.google_api_key else "ollama"
        if self.primary_provider == "google":
            return "google" if self.google_api_key else "ollama"
        if self.primary_provider == "ollama":
            return "ollama"
        # auto: prefer OpenRouter, then Google, then Ollama
        if self.openrouter_configured():
            return "openrouter"
        if self.google_api_key:
            return "google"
        return "ollama"


@dataclass
class AppConfig:
    symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "SPY"])
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    swing: SwingConfig = field(default_factory=SwingConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    briefing: BriefingConfig = field(default_factory=BriefingConfig)
    journal_context: JournalContextConfig = field(default_factory=JournalContextConfig)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    ai_exit: AIExitConfig = field(default_factory=AIExitConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    journal_db_path: str = "data/trades.db"
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_feed: str = "iex"
    finnhub_api_key: str = ""


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or PROJECT_ROOT / "config" / "settings.yaml"
    raw: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    strategy_raw = raw.get("strategy", {})
    swing_raw = raw.get("swing", {})
    session_raw = raw.get("session", {})
    risk_raw = raw.get("risk", {})
    execution_raw = raw.get("execution", {})
    briefing_raw = raw.get("briefing", {})
    journal_context_raw = raw.get("journal_context", {})
    screener_raw = raw.get("screener", {})
    research_raw = raw.get("research", {})
    ai_exit_raw = raw.get("ai_exit", {})
    llm_raw = raw.get("llm", {})
    journal_raw = raw.get("journal", {})

    strategy = StrategyConfig(**{k: strategy_raw[k] for k in StrategyConfig.__dataclass_fields__ if k in strategy_raw})
    hard_stop = ai_exit_raw.get("hard_stop_loss_pct", strategy.stop_loss_pct)

    _openrouter_fallbacks_env = os.getenv("OPENROUTER_FALLBACK_MODELS", "")
    if _openrouter_fallbacks_env:
        openrouter_fallback_models = [m.strip() for m in _openrouter_fallbacks_env.split(",") if m.strip()]
    else:
        openrouter_fallback_models = list(llm_raw.get("openrouter_fallback_models", []))

    _openrouter_reasoning_env = os.getenv("OPENROUTER_REASONING")
    if _openrouter_reasoning_env is not None:
        openrouter_reasoning = _openrouter_reasoning_env.strip().lower() in ("1", "true", "yes", "on")
    else:
        openrouter_reasoning = bool(llm_raw.get("openrouter_reasoning", False))

    llm = LLMConfig(
        enabled=llm_raw.get("enabled", True),
        primary_provider=os.getenv("LLM_PRIMARY_PROVIDER", llm_raw.get("primary_provider", "auto")),
        confidence_threshold=llm_raw.get("confidence_threshold", 0.7),
        timeout_seconds=llm_raw.get("timeout_seconds", 3.0),
        watchlist_interval_minutes=llm_raw.get("watchlist_interval_minutes", 10),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        google_model=os.getenv("GOOGLE_MODEL", llm_raw.get("google_model", "gemma-4-31b-it")),
        google_thinking_level=os.getenv(
            "GOOGLE_THINKING_LEVEL",
            llm_raw.get("google_thinking_level", "MINIMAL"),
        ),
        google_rpm_limit=int(llm_raw.get("google_rpm_limit", os.getenv("GOOGLE_RPM_LIMIT", "12"))),
        ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "phi4-mini"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            llm_raw.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
        ),
        openrouter_model=os.getenv("OPENROUTER_MODEL", llm_raw.get("openrouter_model", "")),
        openrouter_fallback_models=openrouter_fallback_models,
        openrouter_reasoning=openrouter_reasoning,
        openclaw_gateway_url=os.getenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789"),
        openclaw_model=os.getenv("OPENCLAW_MODEL", "gemma4:cloud"),
        alert_channel=os.getenv("OPENCLAW_ALERT_CHANNEL", "telegram"),
        alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL", ""),
    )

    journal_path = journal_raw.get("db_path", "data/trades.db")
    if not Path(journal_path).is_absolute():
        journal_path = str(PROJECT_ROOT / journal_path)

    ai_exit_fields = {k: ai_exit_raw[k] for k in AIExitConfig.__dataclass_fields__ if k in ai_exit_raw}
    ai_exit_fields.setdefault("hard_stop_loss_pct", hard_stop)

    return AppConfig(
        symbols=raw.get("symbols", ["AAPL", "MSFT", "SPY"]),
        strategy=strategy,
        swing=SwingConfig(**{k: swing_raw[k] for k in SwingConfig.__dataclass_fields__ if k in swing_raw}),
        session=SessionConfig(**{k: session_raw[k] for k in SessionConfig.__dataclass_fields__ if k in session_raw}),
        risk=RiskConfig(**{k: risk_raw[k] for k in RiskConfig.__dataclass_fields__ if k in risk_raw}),
        execution=ExecutionConfig(**{k: execution_raw[k] for k in ExecutionConfig.__dataclass_fields__ if k in execution_raw}),
        briefing=BriefingConfig(**{k: briefing_raw[k] for k in BriefingConfig.__dataclass_fields__ if k in briefing_raw}),
        journal_context=JournalContextConfig(**{k: journal_context_raw[k] for k in JournalContextConfig.__dataclass_fields__ if k in journal_context_raw}),
        screener=ScreenerConfig(**{k: screener_raw[k] for k in ScreenerConfig.__dataclass_fields__ if k in screener_raw}),
        research=ResearchConfig(**{k: research_raw[k] for k in ResearchConfig.__dataclass_fields__ if k in research_raw}),
        ai_exit=AIExitConfig(**ai_exit_fields),
        llm=llm,
        journal_db_path=journal_path,
        alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        alpaca_data_feed=os.getenv("ALPACA_DATA_FEED", "iex"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
    )
