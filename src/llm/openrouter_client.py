"""OpenRouter client (OpenAI-compatible) for structured LLM tasks.

Talks to OpenRouter's /chat/completions endpoint. Supports:
- Bearer auth (required by OpenRouter)
- Server-side model fallbacks via the ``models`` array
- A reasoning toggle so "thinking" models (e.g. NVIDIA Nemotron 3 Ultra)
  can be run with reasoning off for fast, clean JSON — or on for deeper
  analysis. Either way we read ``content`` and fall back to ``reasoning``
  so thinking models never silently return empty output.
"""

from __future__ import annotations

import asyncio
import json
import logging

import aiohttp

from src.config import LLMConfig
from src.llm.ollama_client import (
    OllamaClient,
    PremarketBriefing,
    ScreenerRanking,
    WatchlistRanking,
)
from src.llm.prompts import (
    PREMARKET_BRIEFING_PROMPT,
    SCREENER_RANK_PROMPT,
    SWING_SCREENER_RANK_PROMPT,
    SWING_WATCHLIST_RANK_PROMPT,
    WATCHLIST_RANK_PROMPT,
)
from src.logging_sanitize import sanitize_log_message

logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES = frozenset({500, 502, 503, 504})
_MAX_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 3.0


def _safe_error_message(exc: BaseException) -> str:
    return sanitize_log_message(str(exc) or "(no message)")


class OpenRouterClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = config.openrouter_api_key
        self.base_url = config.openrouter_base_url.rstrip("/")
        self.model = config.openrouter_model
        self.fallback_models = list(config.openrouter_fallback_models)

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional attribution headers (used by OpenRouter for rankings).
            "HTTP-Referer": "https://github.com/AI_Stock_portfolio",
            "X-Title": "AI Stock Portfolio",
        }

    def _payload(self, prompt: str) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "stream": False,
            # Toggle "thinking" for reasoning models. Off => fast, clean JSON.
            "reasoning": {"enabled": self.config.openrouter_reasoning},
        }
        if self.fallback_models:
            # Server-side model fallback: OpenRouter tries each in order if the
            # prior one errors or is rate-limited (counts as a single request).
            payload["models"] = [self.model, *self.fallback_models]
        return payload

    def _tools_payload(self, messages: list[dict], tools: list[dict]) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "stream": False,
            "tools": tools,
            # Force a tool call every turn: the only way a ReAct loop ends is
            # via the submit_decision tool, never bare free-text content.
            "tool_choice": "required",
            "reasoning": {"enabled": self.config.openrouter_reasoning},
        }
        if self.fallback_models:
            payload["models"] = [self.model, *self.fallback_models]
        return payload

    @staticmethod
    def _extract_message_text(data: dict) -> str:
        """Return the assistant text, tolerating thinking-model shapes.

        Some reasoning models put the answer in ``content``; others leave
        ``content`` null and stash output in ``reasoning`` / ``reasoning_details``.
        """
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content:
            return content
        reasoning = message.get("reasoning")
        if reasoning:
            return reasoning
        for detail in message.get("reasoning_details") or []:
            text = detail.get("text") or detail.get("content")
            if text:
                return text
        return ""

    async def _post_chat(self, payload: dict) -> dict:
        """POST to /chat/completions with retry/timeout handling; returns the parsed JSON body."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        url = f"{self.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=15,
            sock_read=self.config.timeout_seconds,
        )

        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=self._headers(),
                        json=payload,
                        timeout=timeout,
                    ) as resp:
                        if resp.status >= 400:
                            try:
                                body = await resp.text()
                            except Exception:
                                body = ""
                            detail = (
                                sanitize_log_message(body[:300].replace("\n", " ").strip())
                                if body
                                else ""
                            )
                            # 429 (rate limit) is not retried here: failed
                            # free-tier requests still count toward the daily
                            # quota, so we let the provider chain move on.
                            if resp.status in _RETRYABLE_STATUSES and attempt < _MAX_ATTEMPTS - 1:
                                last_error = aiohttp.ClientResponseError(
                                    resp.request_info,
                                    resp.history,
                                    status=resp.status,
                                    message=f"OpenRouter returned {resp.status}",
                                )
                                logger.warning(
                                    "OpenRouter %s on attempt %d/%d, retrying %s in %.0fs — %s",
                                    resp.status,
                                    attempt + 1,
                                    _MAX_ATTEMPTS,
                                    self.model,
                                    _RETRY_DELAY_SECONDS,
                                    detail or "(no error body)",
                                )
                                await asyncio.sleep(_RETRY_DELAY_SECONDS)
                                continue
                            raise aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                                message=f"OpenRouter returned {resp.status}"
                                + (f": {detail}" if detail else ""),
                            )
                        return await resp.json()
            except (asyncio.TimeoutError, TimeoutError) as e:
                last_error = e
                if attempt < _MAX_ATTEMPTS - 1:
                    logger.warning(
                        "OpenRouter timeout on attempt %d/%d, retrying %s in %.0fs",
                        attempt + 1,
                        _MAX_ATTEMPTS,
                        self.model,
                        _RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenRouter request failed")

    async def _chat(self, prompt: str) -> str:
        data = await self._post_chat(self._payload(prompt))
        text = self._extract_message_text(data)
        if not text:
            raise ValueError("OpenRouter returned empty content")
        return text

    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Send a tool-calling turn; returns the raw assistant message dict (content + tool_calls)."""
        data = await self._post_chat(self._tools_payload(messages, tools))
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("OpenRouter returned no choices")
        return choices[0].get("message") or {}

    @staticmethod
    def _is_rate_limited(exc: BaseException) -> bool:
        return isinstance(exc, aiohttp.ClientResponseError) and exc.status == 429

    async def rank_watchlist(
        self, context: dict, *, swing: bool = False
    ) -> WatchlistRanking | None:
        template = SWING_WATCHLIST_RANK_PROMPT if swing else WATCHLIST_RANK_PROMPT
        prompt = template.format(context=json.dumps(context, indent=2))
        for attempt in range(2):
            try:
                raw = await self._chat(prompt)
                parsed = OllamaClient._extract_json(raw)
                return WatchlistRanking.model_validate(parsed)
            except Exception as e:
                logger.warning(
                    "OpenRouter rank_watchlist attempt %d failed: %s: %s",
                    attempt + 1,
                    type(e).__name__,
                    _safe_error_message(e),
                )
                if self._is_rate_limited(e):
                    return None
        return None

    async def premarket_briefing(self, context: dict) -> PremarketBriefing | None:
        prompt = PREMARKET_BRIEFING_PROMPT.format(
            symbols=", ".join(context.get("symbols", [])),
            news=json.dumps(context.get("news", []), indent=2),
            keyword_flags=json.dumps(context.get("keyword_flags", {}), indent=2),
        )
        for attempt in range(2):
            try:
                raw = await self._chat(prompt)
                parsed = OllamaClient._extract_json(raw)
                return PremarketBriefing.model_validate(parsed)
            except Exception as e:
                logger.warning(
                    "OpenRouter premarket_briefing attempt %d failed: %s: %s",
                    attempt + 1,
                    type(e).__name__,
                    _safe_error_message(e),
                )
                if self._is_rate_limited(e):
                    return None
        return None

    @staticmethod
    def _trim_screener_context(context: dict) -> dict:
        trimmed: list[dict] = []
        for c in context.get("candidates", []):
            row = dict(c)
            headline = row.get("headline")
            if isinstance(headline, str) and len(headline) > 120:
                row["headline"] = headline[:117] + "..."
            trimmed.append(row)
        return {**context, "candidates": trimmed}

    async def screener_rank(
        self, context: dict, *, swing: bool = False
    ) -> ScreenerRanking | None:
        trimmed = self._trim_screener_context(context)
        template = SWING_SCREENER_RANK_PROMPT if swing else SCREENER_RANK_PROMPT
        prompt = template.format(
            slots=trimmed.get("slots", 3),
            candidates=json.dumps(trimmed.get("candidates", []), indent=2),
        )
        for attempt in range(2):
            try:
                raw = await self._chat(prompt)
                parsed = OllamaClient._extract_json(raw)
                return ScreenerRanking.model_validate(parsed)
            except Exception as e:
                logger.warning(
                    "OpenRouter screener_rank attempt %d failed: %s: %s",
                    attempt + 1,
                    type(e).__name__,
                    _safe_error_message(e),
                )
                if self._is_rate_limited(e):
                    return None
        return None
