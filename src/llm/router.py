"""LLM router: Google AI Studio primary (optional), Ollama fallback, OpenClaw alerts."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from src.config import LLMConfig
from src.llm.google_client import GoogleClient
from src.llm.ollama_client import (
    ExitAdvisorDecision,
    OllamaClient,
    PremarketBriefing,
    ScreenerRanking,
    SwingReviewDecision,
    TradeVetoDecision,
    WatchlistRanking,
)
from src.llm.openclaw_client import OpenClawClient
from src.llm.openrouter_client import OpenRouterClient

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

    async def trade_veto(self, context: dict, swing: bool = False) -> tuple[TradeVetoDecision | None, str]:
        if not self.config.enabled:
            return TradeVetoDecision(action="approve", confidence=1.0, reason="llm_disabled"), "none"

        if self._ollama_healthy is None and self._primary not in _REMOTE_PRIMARIES:
            await self.check_health()

        fns = {
            "openrouter": lambda: self.openrouter.trade_veto(context, swing=swing),
            "google": lambda: self.google.trade_veto(context, swing=swing),
            "ollama": lambda: self.ollama.trade_veto(context, swing=swing),
        }

        best: TradeVetoDecision | None = None
        best_source = "none"

        for name, available in self._provider_chain():
            if not available or name not in fns:
                continue
            decision = await fns[name]()
            if decision is None:
                continue
            if decision.confidence >= self.config.confidence_threshold:
                return decision, name
            if best is None or decision.confidence > best.confidence:
                best = decision
                best_source = name

        if best is not None:
            return best, best_source

        logger.info("LLM unavailable — fail-safe reject")
        return (
            TradeVetoDecision(action="reject", confidence=0.0, reason="llm_unavailable"),
            "fail_safe",
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

    async def complete_json(self, prompt: str) -> dict | None:
        """One free-form prompt -> parsed JSON object, with provider fallback (used by ReAct)."""
        if self._ollama_healthy is None and self._primary not in _REMOTE_PRIMARIES:
            await self.check_health()

        def _parsed(chat: Callable[[str], Awaitable[str]], name: str) -> Callable[[], Awaitable[dict | None]]:
            async def call() -> dict | None:
                for attempt in range(2):
                    try:
                        parsed = OllamaClient._extract_json(await chat(prompt))
                    except Exception as e:
                        logger.warning(
                            "%s complete_json attempt %d failed: %s", name, attempt + 1, type(e).__name__
                        )
                        if OpenRouterClient._is_rate_limited(e):
                            return None
                        continue
                    if isinstance(parsed, dict):
                        return parsed
                return None

            return call

        result, _ = await self._first_result(
            {
                "openrouter": _parsed(self.openrouter._chat, "OpenRouter"),
                "google": _parsed(self.google._generate, "Google"),
                "ollama": _parsed(self.ollama._chat, "Ollama"),
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

    async def exit_advisor(self, context: dict) -> tuple[ExitAdvisorDecision, str]:
        """Fail-safe: sell if LLM unavailable or low confidence."""
        sell = ExitAdvisorDecision(action="sell", confidence=0.0, reason="llm_unavailable_fail_safe")

        if not self.config.enabled:
            return sell, "none"

        if self._ollama_healthy is None and self._primary not in _REMOTE_PRIMARIES:
            await self.check_health()

        result, source = await self._first_result(
            {
                "openrouter": lambda: self.openrouter.exit_advisor(context),
                "google": lambda: self.google.exit_advisor(context),
                "ollama": lambda: self.ollama.exit_advisor(context),
            }
        )
        if result is None:
            logger.info("LLM exit_advisor unavailable — fail-safe sell")
            return sell, "fail_safe"

        decision = result  # type: ignore[assignment]
        if decision.confidence < self.config.confidence_threshold:
            return (
                ExitAdvisorDecision(
                    action="sell",
                    confidence=decision.confidence,
                    reason=f"low_confidence: {decision.reason}",
                ),
                source,
            )
        return decision, source

    async def swing_review(self, context: dict) -> tuple[SwingReviewDecision, str]:
        """Morning review: low-confidence exits become hold (hard stops still protect)."""
        hold_default = SwingReviewDecision(
            action="hold",
            confidence=0.0,
            reason="llm_unavailable_fail_safe_hold",
        )

        if not self.config.enabled:
            return hold_default, "none"

        if self._ollama_healthy is None and self._primary not in _REMOTE_PRIMARIES:
            await self.check_health()

        result, source = await self._first_result(
            {
                "openrouter": lambda: self.openrouter.swing_review(context),
                "google": lambda: self.google.swing_review(context),
                "ollama": lambda: self.ollama.swing_review(context),
            }
        )
        if result is None:
            logger.info("LLM swing_review unavailable — fail-safe hold")
            return hold_default, "fail_safe"

        decision = result  # type: ignore[assignment]
        if decision.confidence < self.config.confidence_threshold:
            # Do NOT force exit on low confidence — that closed EQNR while the
            # model reason still sounded bullish. Prefer hold; trail is skipped.
            if decision.action in ("exit", "trail"):
                logger.info(
                    "LLM swing_review low confidence (%.2f < %.2f) — overriding %s -> hold",
                    decision.confidence,
                    self.config.confidence_threshold,
                    decision.action,
                )
                return (
                    SwingReviewDecision(
                        action="hold",
                        confidence=decision.confidence,
                        reason=f"low_confidence_hold: {decision.reason}",
                    ),
                    source,
                )
        return decision, source

    async def alert(self, title: str, message: str) -> None:
        await self.openclaw.send_alert(title, message)


def _keyword_avoid_list(context: dict) -> list[str]:
    flags = context.get("keyword_flags", {})
    return [sym for sym, hits in flags.items() if hits]
