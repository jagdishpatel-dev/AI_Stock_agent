"""ReAct (Reason + Act) tool-calling loop for trading decisions.

Runs a bounded loop of OpenRouter tool-calling turns: the model can call
read-only tools to gather information, then must call submit_decision
exactly once to finish. tool_choice="required" makes every turn end in a
tool call, so a decision is never accepted from free-text content — only
from a validated submit_decision call. If the iteration cap is hit without
one, or OpenRouter is unavailable/errors, the loop fails safe to
`fail_safe_action` — it never fails open into a trade.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable

from pydantic import ValidationError

from src.llm.openrouter_client import OpenRouterClient
from src.llm.react_tools import SUBMIT_DECISION_TOOL, ReactDecision
from src.logging_sanitize import sanitize_log_message

logger = logging.getLogger(__name__)


def _fail_safe(action: str, reason: str) -> ReactDecision:
    return ReactDecision(action=action, confidence=0.0, reasoning=reason)


def _tool_result_message(tool_call_id: str, result: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result),
    }


async def run_react_loop(
    openrouter: OpenRouterClient,
    *,
    system_prompt: str,
    user_prompt: str,
    read_tools: list[dict],
    tool_dispatch: dict[str, Callable[[], Awaitable[dict]]],
    allowed_actions: set[str],
    fail_safe_action: str,
    max_iterations: int = 5,
) -> tuple[ReactDecision, list[dict]]:
    """Run the ReAct loop.

    Returns (decision, trace); trace is an ordered list of
    {thought, action, arguments, observation} dicts, one per tool call made,
    suitable for logging verbatim to the journal for auditability.
    """
    trace: list[dict] = []

    if not openrouter.configured:
        return _fail_safe(fail_safe_action, "openrouter_unconfigured"), trace

    tools = [*read_tools, SUBMIT_DECISION_TOOL]
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(max_iterations):
        try:
            message = await openrouter.chat_with_tools(messages, tools)
        except Exception as e:
            logger.warning(
                "ReAct loop: OpenRouter call failed: %s: %s",
                type(e).__name__,
                sanitize_log_message(str(e) or "(no message)"),
            )
            return _fail_safe(fail_safe_action, "openrouter_error"), trace

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            # tool_choice="required" should make this unreachable; nudge and retry.
            messages.append({"role": "assistant", "content": message.get("content") or ""})
            messages.append(
                {
                    "role": "user",
                    "content": "You must call a tool. Call submit_decision when you are ready to decide.",
                }
            )
            continue

        messages.append(
            {"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls}
        )

        submit_call = next(
            (tc for tc in tool_calls if tc.get("function", {}).get("name") == "submit_decision"),
            None,
        )

        if submit_call is not None:
            raw_args = submit_call.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            trace.append(
                {
                    "thought": message.get("content"),
                    "action": "submit_decision",
                    "arguments": args,
                    "observation": None,
                }
            )
            try:
                decision = ReactDecision.model_validate(args)
            except ValidationError as e:
                messages.append(
                    _tool_result_message(submit_call["id"], {"error": f"invalid arguments: {e}"})
                )
                continue
            if decision.action not in allowed_actions:
                messages.append(
                    _tool_result_message(
                        submit_call["id"],
                        {"error": f"action must be one of {sorted(allowed_actions)}"},
                    )
                )
                continue
            return decision, trace

        # No submit_decision this turn — execute the read-only tool calls and continue.
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            impl = tool_dispatch.get(name)
            if impl is None:
                result: dict = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = await impl()
                except Exception as e:
                    result = {"error": sanitize_log_message(str(e) or type(e).__name__)}
            trace.append(
                {"thought": message.get("content"), "action": name, "arguments": {}, "observation": result}
            )
            messages.append(_tool_result_message(tc["id"], result))

    logger.info(
        "ReAct loop hit iteration cap (%d) without submit_decision — fail-safe to %s",
        max_iterations,
        fail_safe_action,
    )
    return _fail_safe(fail_safe_action, "iteration_cap_exhausted"), trace
