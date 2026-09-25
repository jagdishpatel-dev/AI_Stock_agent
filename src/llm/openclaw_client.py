"""OpenClaw gateway client for fallback LLM and alert dispatch."""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

from src.config import LLMConfig
from src.llm.ollama_client import (
    OllamaClient,
    PremarketBriefing,
    ScreenerRanking,
    WatchlistRanking,
)
from src.llm.prompts import (
    ALERT_SUMMARY_PROMPT,
    PREMARKET_BRIEFING_PROMPT,
    SCREENER_RANK_PROMPT,
    SWING_SCREENER_RANK_PROMPT,
    SWING_WATCHLIST_RANK_PROMPT,
    WATCHLIST_RANK_PROMPT,
)

logger = logging.getLogger(__name__)


class OpenClawClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.gateway_url = config.openclaw_gateway_url.rstrip("/")
        self.model = config.openclaw_model

    async def _agent_chat(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds * 2)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.gateway_url}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            ) as resp:
                if resp.status == 404:
                    async with session.post(
                        f"{self.gateway_url}/api/chat",
                        json={"model": self.model, "messages": payload["messages"], "stream": False},
                        timeout=timeout,
                    ) as fallback_resp:
                        fallback_resp.raise_for_status()
                        data = await fallback_resp.json()
                        if "message" in data:
                            return data["message"].get("content", "")
                        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                resp.raise_for_status()
                data = await resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def rank_watchlist(
        self, context: dict, *, swing: bool = False
    ) -> WatchlistRanking | None:
        template = SWING_WATCHLIST_RANK_PROMPT if swing else WATCHLIST_RANK_PROMPT
        prompt = template.format(context=json.dumps(context, indent=2))
        try:
            raw = await self._agent_chat(prompt)
            from src.llm.ollama_client import OllamaClient

            parsed = OllamaClient._extract_json(raw)
            return WatchlistRanking.model_validate(parsed)
        except Exception as e:
            logger.warning("OpenClaw rank_watchlist failed: %s", e)
            return None

    async def premarket_briefing(self, context: dict) -> PremarketBriefing | None:
        prompt = PREMARKET_BRIEFING_PROMPT.format(
            symbols=", ".join(context.get("symbols", [])),
            news=json.dumps(context.get("news", []), indent=2),
            keyword_flags=json.dumps(context.get("keyword_flags", {}), indent=2),
        )
        try:
            raw = await self._agent_chat(prompt)
            from src.llm.ollama_client import OllamaClient

            parsed = OllamaClient._extract_json(raw)
            return PremarketBriefing.model_validate(parsed)
        except Exception as e:
            logger.warning("OpenClaw premarket_briefing failed: %s", e)
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
            raw = await self._agent_chat(prompt)
            from src.llm.ollama_client import OllamaClient

            parsed = OllamaClient._extract_json(raw)
            return ScreenerRanking.model_validate(parsed)
        except Exception as e:
            logger.warning("OpenClaw screener_rank failed: %s", e)
            return None

    async def send_alert(self, title: str, message: str) -> bool:
        text = f"**{title}**\n{message}"
        if self.config.alert_webhook_url:
            return await self._send_webhook(text)
        return await self._send_gateway_message(text)

    async def _send_webhook(self, text: str) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.alert_webhook_url,
                    json={"text": text},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status < 400
        except Exception as e:
            logger.warning("Webhook alert failed: %s", e)
            return False

    async def _send_gateway_message(self, text: str) -> bool:
        payload: dict[str, Any] = {
            "channel": self.config.alert_channel,
            "message": text,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.gateway_url}/api/message",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status < 400:
                        return True
                    logger.debug("Gateway message endpoint returned %s", resp.status)
        except Exception as e:
            logger.debug("Gateway alert failed (configure webhook or OpenClaw channels): %s", e)
        return False

    async def daily_summary(self, context: dict) -> str:
        prompt = ALERT_SUMMARY_PROMPT.format(context=json.dumps(context, indent=2))
        try:
            return await self._agent_chat(prompt)
        except Exception:
            return json.dumps(context, indent=2)
