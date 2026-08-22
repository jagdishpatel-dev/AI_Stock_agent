"""LLM router: Google AI Studio primary (optional), Ollama fallback, OpenClaw alerts."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from src.config import LLMConfig
from src.llm.google_client import GoogleClient
from src.llm.ollama_client import (
    OllamaClient,
    PremarketBriefing,
    ScreenerRanking,
    WatchlistRanking,
)
from src.llm.openclaw_client import OpenClawClient
from src.llm.openrouter_client import OpenRouterClient
from src.llm.prompts import (
    REACT_ENTRY_SYSTEM_PROMPT,
    REACT_EXIT_ADVISOR_SYSTEM_PROMPT,
    REACT_SWING_ENTRY_SYSTEM_PROMPT,
    REACT_SWING_REVIEW_SYSTEM_PROMPT,
)
from src.llm.react_engine import run_react_loop
from src.llm.react_tools import ALL_READ_TOOLS, ReactDecision, ToolContext, build_tool_dispatch

logger = logging.getLogger(__name__)

# Primaries that don't require a local Ollama health probe at call time.
_REMOTE_PRIMARIES = ("google", "openrouter")


class LLMRouter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.google = GoogleClient(config)
        self.ollama = OllamaClient(config)
        self.openrouter = OpenRouterClient(config)
        self.openclaw = OpenClawClient(config)
        self._ollama_healthy: bool | None = None
        self._primary = config.resolved_primary()

    async def check_health(self) -> bool:
        self._ollama_healthy = await self.ollama.health_check()
        if self._primary == "openrouter" and self.openrouter.configured:
            return True
        if self._primary == "google" and self.google.configured:
            return True
        return bool(self._ollama_healthy)

    def _provider_chain(self) -> list[tuple[str, bool]]:
        """Ordered list of (name, is_available) for LLM providers."""
        chain: list[tuple[str, bool]] = []
        if self._primary == "openrouter" and self.openrouter.configured:
            chain.append(("openrouter", True))
            if self.google.configured:
                chain.append(("google", True))
            chain.append(("ollama", self._ollama_healthy is True))
        elif self._primary == "google" and self.google.configured:
            chain.append(("google", True))
            if self.openrouter.configured:
                chain.append(("openrouter", True))
            chain.append(("ollama", self._ollama_healthy is True))
        else:
            chain.append(("ollama", self._ollama_healthy is not False))
            if self.openrouter.configured:
                chain.append(("openrouter", True))
            if self.google.configured:
                chain.append(("google", True))
        return chain

    async def _first_result(
        self,
        fns: dict[str, Callable[[], Awaitable[object | None]]],
    ) -> tuple[object | None, str]:
        for name, available in self._provider_chain():
            if not available or name not in fns:
                continue
            result = await fns[name]()
            if result is not None:
                return result, name
        return None, "none"

    async def trade_veto_react(
        self,
        symbol: str,
        signal_reason: str,
        tool_ctx: ToolContext,
        *,
        swing: bool = False,
    ) -> tuple[ReactDecision, list[dict]]:
        """ReAct entry veto: approve (buy) or reject (hold) a rule-generated BUY signal.

        OpenRouter only — no Google/Ollama fallback. Fails safe to "hold" (reject)
        if OpenRouter is unconfigured, errors, or the loop exhausts its iteration cap.
        """
        cfg = tool_ctx.config
        template = REACT_SWING_ENTRY_SYSTEM_PROMPT if swing else REACT_ENTRY_SYSTEM_PROMPT
        system_prompt = template.format(
            symbol=symbol,
            signal_reason=signal_reason,
            max_spread_pct=cfg.execution.max_spread_pct,
            min_trades_for_veto=cfg.journal_context.min_trades_for_veto,
            min_win_rate=cfg.journal_context.min_win_rate,
            confidence_threshold=self.config.confidence_threshold,
        )
        return await run_react_loop(
            self.openrouter,
            system_prompt=system_prompt,
            user_prompt=f"Evaluate this candidate BUY signal for {symbol} and decide.",
            read_tools=ALL_READ_TOOLS,
            tool_dispatch=build_tool_dispatch(tool_ctx),
            allowed_actions={"buy", "hold"},
            fail_safe_action="hold",
            max_iterations=self.config.react_max_iterations,
        )

    async def rank_watchlist(
        self, context: dict, *, swing: bool = False
    ) -> WatchlistRanking | None:
        if not self.config.enabled:
            return None
        if self._ollama_healthy is None and self._primary not in _REMOTE_PRIMARIES:
            await self.check_health()
        result, _ = await self._first_result(
            {
                "openrouter": lambda: self.openrouter.rank_watchlist(context, swing=swing),
                "google": lambda: self.google.rank_watchlist(context, swing=swing),
                "ollama": lambda: self.ollama.rank_watchlist(context, swing=swing),
            }
        )
        return result  # type: ignore[return-value]

    async def briefing_decision(self, context: dict) -> PremarketBriefing:
        if self._ollama_healthy is None and self._primary not in _REMOTE_PRIMARIES:
            await self.check_health()
        briefing, _ = await self._first_result(
            {
                "openrouter": lambda: self.openrouter.premarket_briefing(context),
                "google": lambda: self.google.premarket_briefing(context),
                "ollama": lambda: self.ollama.premarket_briefing(context),
            }
        )
        if briefing is not None:
            return briefing  # type: ignore[return-value]

        return PremarketBriefing(
            avoid=_keyword_avoid_list(context),
            caution=[],
            reason="LLM unavailable — using keyword-only detection",
        )

    async def screener_rank(
        self, context: dict, *, swing: bool = False
    ) -> ScreenerRanking | None:
        if self._ollama_healthy is None and self._primary not in _REMOTE_PRIMARIES:
            await self.check_health()
        result, _ = await self._first_result(
            {
                "openrouter": lambda: self.openrouter.screener_rank(context, swing=swing),
                "google": lambda: self.google.screener_rank(context, swing=swing),
                "ollama": lambda: self.ollama.screener_rank(context, swing=swing),
            }
        )
        return result  # type: ignore[return-value]

    async def exit_advisor_react(self, zone: str, tool_ctx: ToolContext) -> tuple[ReactDecision, list[dict]]:
        """ReAct intraday exit advisor: hold or sell an open scalp position.

        OpenRouter only — no Google/Ollama fallback. Fails safe to "sell" (matches
        the original single-shot behavior) if OpenRouter is unconfigured, errors,
        or the loop exhausts its iteration cap.
        """
        ai = tool_ctx.config.ai_exit
        system_prompt = REACT_EXIT_ADVISOR_SYSTEM_PROMPT.format(
            symbol=tool_ctx.symbol,
            zone=zone,
            hard_stop_pct=ai.hard_stop_loss_pct,
            min_take_profit_pct=ai.min_take_profit_pct,
            max_target_pct=ai.max_target_pct,
            max_hold_minutes=ai.max_hold_minutes,
            max_loss_hold_minutes=ai.max_loss_hold_minutes,
        )
        decision, trace = await run_react_loop(
            self.openrouter,
            system_prompt=system_prompt,
            user_prompt=f"An open position in {tool_ctx.symbol} is in the {zone} zone. Decide whether to sell or hold.",
            read_tools=ALL_READ_TOOLS,
            tool_dispatch=build_tool_dispatch(tool_ctx),
            allowed_actions={"sell", "hold"},
            fail_safe_action="sell",
            max_iterations=self.config.react_max_iterations,
        )
        if decision.confidence < self.config.confidence_threshold and decision.action == "hold":
            decision = ReactDecision(
                action="sell",
                confidence=decision.confidence,
                reasoning=f"low_confidence: {decision.reasoning}",
            )
        return decision, trace

    async def swing_review_react(self, tool_ctx: ToolContext) -> tuple[ReactDecision, list[dict]]:
        """ReAct morning swing review: hold, sell (exit), or trail an open swing position.

        OpenRouter only — no Google/Ollama fallback. Fails safe to "hold" (hard
        stops still protect) if OpenRouter is unconfigured, errors, or the loop
        exhausts its iteration cap.
        """
        swing = tool_ctx.config.swing
        system_prompt = REACT_SWING_REVIEW_SYSTEM_PROMPT.format(
            symbol=tool_ctx.symbol,
            days_held=tool_ctx.days_held,
            take_profit_pct=swing.take_profit_pct,
            trail_stop_pct=swing.trailing_stop_pct,
            hard_stop_pct=swing.hard_stop_pct,
            max_hold_days=swing.max_hold_days,
        )
        decision, trace = await run_react_loop(
            self.openrouter,
            system_prompt=system_prompt,
            user_prompt=f"Review the open swing position in {tool_ctx.symbol} (day {tool_ctx.days_held}) and decide.",
            read_tools=ALL_READ_TOOLS,
            tool_dispatch=build_tool_dispatch(tool_ctx),
            allowed_actions={"sell", "hold", "trail"},
            fail_safe_action="hold",
            max_iterations=self.config.react_max_iterations,
        )
        if decision.confidence < self.config.confidence_threshold and decision.action in ("sell", "trail"):
            # Do NOT force exit on low confidence — prefer hold; trail is skipped too.
            logger.info(
                "swing_review_react low confidence (%.2f < %.2f) — overriding %s -> hold",
                decision.confidence,
                self.config.confidence_threshold,
                decision.action,
            )
            decision = ReactDecision(
                action="hold",
                confidence=decision.confidence,
                reasoning=f"low_confidence_hold: {decision.reasoning}",
            )
        return decision, trace

    async def alert(self, title: str, message: str) -> None:
        await self.openclaw.send_alert(title, message)


def _keyword_avoid_list(context: dict) -> list[str]:
    flags = context.get("keyword_flags", {})
    return [sym for sym, hits in flags.items() if hits]
