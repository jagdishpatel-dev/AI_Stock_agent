"""Direct Ollama /api/chat client with structured JSON output."""

from __future__ import annotations

import json
import logging
import re

import aiohttp
from pydantic import BaseModel, Field

from src.config import LLMConfig
from src.llm.prompts import (
    PREMARKET_BRIEFING_PROMPT,
    SCREENER_RANK_PROMPT,
    SWING_SCREENER_RANK_PROMPT,
    SWING_WATCHLIST_RANK_PROMPT,
    WATCHLIST_RANK_PROMPT,
)

logger = logging.getLogger(__name__)


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


class OllamaClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.base_url = config.ollama_host.rstrip("/")
        self.model = config.ollama_model
        self.api_key = config.ollama_api_key

    def _headers(self) -> dict[str, str]:
        # Local Ollama needs no auth; Ollama Cloud (https://ollama.com) requires
        # a Bearer token from an OLLAMA_API_KEY (see https://docs.ollama.com/cloud).
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    headers=self._headers(),
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
                headers=self._headers(),
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

