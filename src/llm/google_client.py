"""Google AI Studio / Gemini API client for structured LLM tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import time

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

_GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 10.0


def _safe_error_message(exc: BaseException) -> str:
    return sanitize_log_message(str(exc) or "(no message)")


class GoogleClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = config.google_api_key
        self.model = config.google_model
        self._last_request_at: float = 0.0
        self._rate_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _is_gemma(self) -> bool:
        """Gemma models don't support JSON mode or thinking config."""
        return "gemma" in self.model.lower()

    def _generation_config(self) -> dict:
        config: dict = {"temperature": 0.1}

        # Gemma models on the Generative Language API reject responseMimeType
        # (JSON mode) and thinkingConfig — sending them causes 400/500 errors.
        # Prompts already instruct JSON-only output and _extract_json parses it.
        if self._is_gemma():
            return config

        config["responseMimeType"] = "application/json"
        level = self.config.google_thinking_level.strip()
        if level and level.lower() not in ("off", "none", "disabled"):
            config["thinkingConfig"] = {"thinkingLevel": level.upper()}
        return config

    async def _rate_limit(self) -> None:
        rpm = self.config.google_rpm_limit
        if rpm <= 0:
            return
        min_interval = 60.0 / rpm
        async with self._rate_lock:
            now = time.monotonic()
            wait = min_interval - (now - self._last_request_at)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    async def _generate(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        url = f"{_GOOGLE_API_BASE}/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": self._generation_config(),
        }
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=15,
            sock_read=self.config.timeout_seconds,
        )

        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                await self._rate_limit()
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        params={"key": self.api_key},
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
                            message = f"Google API returned {resp.status}" + (
                                f": {detail}" if detail else ""
                            )
                            err = aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                                message=message,
                            )
                            if resp.status in _RETRYABLE_STATUSES and attempt < _MAX_ATTEMPTS - 1:
                                last_error = err
                                logger.warning(
                                    "Google API %s on attempt %d/%d, retrying %s in %.0fs — %s",
                                    resp.status,
                                    attempt + 1,
                                    _MAX_ATTEMPTS,
                                    self.model,
                                    _RETRY_DELAY_SECONDS,
                                    detail or "(no error body)",
                                )
                                await asyncio.sleep(_RETRY_DELAY_SECONDS)
                                continue
                            raise err

                        data = await resp.json()
            except (asyncio.TimeoutError, TimeoutError) as e:
                last_error = e
                if attempt < _MAX_ATTEMPTS - 1:
                    logger.warning(
                        "Google API timeout on attempt %d/%d, retrying %s in %.0fs",
                        attempt + 1,
                        _MAX_ATTEMPTS,
                        self.model,
                        _RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    continue
                raise
            except aiohttp.ClientResponseError as e:
                last_error = e
                if e.status in _RETRYABLE_STATUSES and attempt < _MAX_ATTEMPTS - 1:
                    logger.warning(
                        "Google API %s on attempt %d/%d, retrying %s in %.0fs",
                        e.status,
                        attempt + 1,
                        _MAX_ATTEMPTS,
                        self.model,
                        _RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    continue
                raise aiohttp.ClientResponseError(
                    e.request_info,
                    e.history,
                    status=e.status,
                    message=sanitize_log_message(e.message or f"Google API returned {e.status}"),
                ) from None
            else:
                candidates = data.get("candidates") or []
                if not candidates:
                    raise ValueError("Google API returned no candidates")
                return self._extract_response_text(candidates[0])

        if last_error is not None:
            raise last_error
        raise RuntimeError("Google API request failed")

    @staticmethod
    def _extract_response_text(candidate: dict) -> str:
        """Return model output, skipping Gemma 4 internal thought parts."""
        parts = candidate.get("content", {}).get("parts") or []
        if not parts:
            raise ValueError("Google API returned empty content")

        for part in reversed(parts):
            if part.get("thought"):
                continue
            text = part.get("text", "")
            if text:
                return text

        for part in reversed(parts):
            text = part.get("text", "")
            if text:
                return text

        raise ValueError("Google API returned empty content")

    async def rank_watchlist(
        self, context: dict, *, swing: bool = False
    ) -> WatchlistRanking | None:
        template = SWING_WATCHLIST_RANK_PROMPT if swing else WATCHLIST_RANK_PROMPT
        prompt = template.format(context=json.dumps(context, indent=2))
        try:
            raw = await self._generate(prompt)
            parsed = OllamaClient._extract_json(raw)
            return WatchlistRanking.model_validate(parsed)
        except Exception as e:
            logger.warning(
                "Google rank_watchlist failed: %s: %s",
                type(e).__name__,
                _safe_error_message(e),
            )
            return None

    async def premarket_briefing(self, context: dict) -> PremarketBriefing | None:
        prompt = PREMARKET_BRIEFING_PROMPT.format(
            symbols=", ".join(context.get("symbols", [])),
            news=json.dumps(context.get("news", []), indent=2),
            keyword_flags=json.dumps(context.get("keyword_flags", {}), indent=2),
        )
        try:
            raw = await self._generate(prompt)
            parsed = OllamaClient._extract_json(raw)
            return PremarketBriefing.model_validate(parsed)
        except Exception as e:
            logger.warning(
                "Google premarket_briefing failed: %s: %s",
                type(e).__name__,
                _safe_error_message(e),
            )
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
                raw = await self._generate(prompt)
                parsed = OllamaClient._extract_json(raw)
                return ScreenerRanking.model_validate(parsed)
            except Exception as e:
                logger.warning(
                    "Google screener_rank attempt %d failed: %s: %s",
                    attempt + 1,
                    type(e).__name__,
                    _safe_error_message(e),
                )
        return None
