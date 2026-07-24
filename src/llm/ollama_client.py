"""Direct Ollama /api/chat client with structured JSON output."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Literal

import aiohttp
from pydantic import BaseModel, Field, ValidationError

from src.config import LLMConfig
from src.llm.prompts import (
    EXIT_ADVISOR_PROMPT,
    PREMARKET_BRIEFING_PROMPT,
    SCREENER_RANK_PROMPT,
    SWING_REVIEW_PROMPT,
    SWING_SCREENER_RANK_PROMPT,
    SWING_VETO_PROMPT,
    SWING_WATCHLIST_RANK_PROMPT,
    TRADE_VETO_PROMPT,
    WATCHLIST_RANK_PROMPT,
)

logger = logging.getLogger(__name__)


class TradeVetoDecision(BaseModel):
    action: Literal["approve", "reject"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class WatchlistRanking(BaseModel):
    ranked: list[str]
    reason: str = ""


class PremarketBriefing(BaseModel):
    avoid: list[str] = Field(default_factory=list)
    caution: list[str] = Field(default_factory=list)
    reason: str = ""


class ScreenerRanking(BaseModel):
    picks: list[str] = Field(default_factory=list)
    reasons: dict[str, str] = Field(default_factory=dict)
    summary: str = ""


class ExitAdvisorDecision(BaseModel):
    action: Literal["hold", "sell"]
    target_pct: float | None = None
    max_hold_minutes: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class SwingReviewDecision(BaseModel):
    action: Literal["hold", "exit", "trail"]
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    new_stop_pct: float | None = None  # tighter trailing stop distance from high
    reason: str = ""


class OllamaClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.base_url = config.ollama_host.rstrip("/")
        self.model = config.ollama_model

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def _chat(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192,
            },
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("message", {}).get("content", "")

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    async def trade_veto(self, context: dict, swing: bool = False) -> TradeVetoDecision | None:
        template = SWING_VETO_PROMPT if swing else TRADE_VETO_PROMPT
        prompt = template.format(context=json.dumps(context, indent=2))
        for attempt in range(2):
            try:
                raw = await self._chat(prompt)
                parsed = self._extract_json(raw)
                return TradeVetoDecision.model_validate(parsed)
            except (asyncio.TimeoutError, aiohttp.ClientError, ValidationError, json.JSONDecodeError) as e:
                logger.warning("Ollama trade_veto attempt %d failed: %s", attempt + 1, e)
        return None

    async def rank_watchlist(
        self, context: dict, *, swing: bool = False
    ) -> WatchlistRanking | None:
        template = SWING_WATCHLIST_RANK_PROMPT if swing else WATCHLIST_RANK_PROMPT
        prompt = template.format(context=json.dumps(context, indent=2))
        try:
            raw = await self._chat(prompt)
            parsed = self._extract_json(raw)
            return WatchlistRanking.model_validate(parsed)
        except Exception as e:
            logger.warning("Ollama rank_watchlist failed: %s", e)
            return None

    async def premarket_briefing(self, context: dict) -> PremarketBriefing | None:
        prompt = PREMARKET_BRIEFING_PROMPT.format(
            symbols=", ".join(context.get("symbols", [])),
            news=json.dumps(context.get("news", []), indent=2),
            keyword_flags=json.dumps(context.get("keyword_flags", {}), indent=2),
        )
        try:
            raw = await self._chat(prompt)
            parsed = self._extract_json(raw)
            return PremarketBriefing.model_validate(parsed)
        except Exception as e:
            logger.warning("Ollama premarket_briefing failed: %s", e)
            return None

    async def screener_rank(
        self, context: dict, *, swing: bool = False
    ) -> ScreenerRanking | None:
        template = SWING_SCREENER_RANK_PROMPT if swing else SCREENER_RANK_PROMPT
        prompt = template.format(
            slots=context.get("slots", 3),
            candidates=json.dumps(context.get("candidates", []), indent=2),
        )
        try:
            raw = await self._chat(prompt)
            parsed = self._extract_json(raw)
            return ScreenerRanking.model_validate(parsed)
        except Exception as e:
            logger.warning("Ollama screener_rank failed: %s", e)
            return None

    async def swing_review(self, context: dict) -> SwingReviewDecision | None:
        swing = context.get("_swing_limits", {})
        prompt = SWING_REVIEW_PROMPT.format(
            days_held=context.get("days_held", 0),
            context=json.dumps({k: v for k, v in context.items() if not k.startswith("_")}, indent=2),
            take_profit_pct=swing.get("take_profit_pct", 2.5),
            trail_stop_pct=swing.get("trailing_stop_pct", 0.5),
            hard_stop_pct=swing.get("hard_stop_pct", 1.5),
            max_hold_days=swing.get("max_hold_days", 5),
        )
        try:
            raw = await self._chat(prompt)
            parsed = self._extract_json(raw)
            return SwingReviewDecision.model_validate(parsed)
        except Exception as e:
            logger.warning("Ollama swing_review failed: %s", e)
            return None

    async def exit_advisor(self, context: dict) -> ExitAdvisorDecision | None:
        ai_exit = context.get("_ai_exit_limits", {})
        prompt = EXIT_ADVISOR_PROMPT.format(
            zone=context.get("zone", "profit"),
            hard_stop_pct=ai_exit.get("hard_stop_loss_pct", 0.12),
            context=json.dumps({k: v for k, v in context.items() if not k.startswith("_")}, indent=2),
            min_take_profit_pct=ai_exit.get("min_take_profit_pct", 0.20),
            max_target_pct=ai_exit.get("max_target_pct", 1.0),
            max_hold_minutes=ai_exit.get("max_hold_minutes", 20),
            max_loss_hold_minutes=ai_exit.get("max_loss_hold_minutes", 8),
        )
        try:
            raw = await self._chat(prompt)
            parsed = self._extract_json(raw)
            return ExitAdvisorDecision.model_validate(parsed)
        except Exception as e:
            logger.warning("Ollama exit_advisor failed: %s", e)
            return None
