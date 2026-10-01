"""LLM prompt templates for trade veto and watchlist ranking."""

TRADE_VETO_PROMPT = """You are a scalping trade risk filter. You may ONLY approve or reject a trade signal.
You cannot suggest new trades. Respond with JSON only, no other text.

Signal context:
{context}

Rules:
- Reject if spread_pct >= max_spread_pct or volume_ratio < 1.2
- Reject if RSI is not clearly oversold for a bounce entry
- Reject if historical_stats shows similar_trades >= 5 and win_rate < 0.4
- Use outcome_cards when present: if recent similar trades are mostly losers (e.g. <=1 win in last 5),
  reject unless this setup is clearly different
- Approve only if confidence >= 0.7

Respond exactly:
{{"action": "approve" or "reject", "confidence": 0.0-1.0, "reason": "brief reason"}}"""

SWING_VETO_PROMPT = """You are a swing/momentum trade risk filter. A rule-based momentum system has ALREADY
generated this BUY signal. You may ONLY approve or reject it — you cannot suggest new trades.
Respond with JSON only, no other text.

The entry system already confirmed a momentum breakout: price above VWAP, EMA fast > EMA slow,
a volume spike, RSI below the overbought ceiling, and an acceptable opening gap. Your job is to
catch obvious risks, NOT to require an oversold/mean-reversion setup.

Signal context:
{context}

Rules:
- This is a MOMENTUM entry. Do NOT reject just because RSI is high or "not oversold" — elevated RSI
  (roughly up to 72) is expected and healthy for a momentum breakout.
- NEVER reject solely for "RSI not oversold" or "waiting for a bounce" — that is the wrong strategy.
- Use daily_price_history and daily_trend when present: recent multi-day pullback with improving
  intraday momentum can be a valid buy; reject if price is in free-fall (many consecutive down days
  with no intraday strength) or far below the period low without reversal signs.
- Reject if spread_pct is present and clearly too wide (>= max_spread_pct).
- Reject if momentum is actually negative: vwap_deviation_pct < 0, or ema_fast <= ema_slow.
- Reject if gap_pct is extreme (> 8%) — gap-and-crap / exhaustion risk.
- Reject if historical_stats shows similar_trades >= 5 and win_rate < 0.4.
- Use outcome_cards when present:
  * Read outcome_cards.summary and the last few cards (pnl, exit_reason, rsi, vwap_dev).
  * If recent similar outcomes are mostly losses (e.g. <=1 win in last 5) or many exited via
    dynamic_stop / hard_stop within the same day, prefer REJECT unless this setup differs clearly.
  * Mention the outcome pattern briefly in your reason when it influences the decision.
- Otherwise APPROVE. On a clean momentum setup with neutral/positive outcome history, approve with
  confidence >= 0.7.

Respond exactly:
{{"action": "approve" or "reject", "confidence": 0.0-1.0, "reason": "brief reason"}}"""

WATCHLIST_RANK_PROMPT = """Rank these symbols for intraday scalping priority based on the context.
Respond with JSON only.

Symbols and metrics:
{context}

Rules:
- Prefer symbols with strong short-term momentum (positive research_change_pct or price above VWAP)
- If data_quality is "partial", use research_price and research_change_pct from Yahoo research
- Do NOT rank symbols last solely because live rsi/vwap_dev/close are null when research metrics are present

Respond exactly:
{{"ranked": ["SYMBOL1", "SYMBOL2", ...], "reason": "brief reason"}}"""

SWING_WATCHLIST_RANK_PROMPT = """Rank these symbols for multi-day SWING / momentum trading priority.
Respond with JSON only. This is NOT intraday scalping — prefer names that can hold 1–5 days.

Symbols and metrics:
{context}

Rules:
- Prefer liquid names with sustained upward momentum (positive vwap_dev or research_change_pct)
- Prefer moderate gaps and healthy volume over extreme one-day spikes that often fade
- If data_quality is "partial", use research_price and research_change_pct from Yahoo research
- Do NOT rank symbols last solely because live rsi/vwap_dev/close are null when research metrics are present
- Deprioritize symbols that look like exhaustion / gap-and-crap candidates

Respond exactly:
{{"ranked": ["SYMBOL1", "SYMBOL2", ...], "reason": "brief reason"}}"""

REACT_WATCHLIST_RULES_SCALP = """- Goal: intraday scalping priority
- Prefer strong short-term momentum (price above VWAP, positive change_pct, rising volume)
- Deprioritize names with risky fresh news or that already sit on the pre-market avoid list"""

REACT_WATCHLIST_RULES_SWING = """- Goal: multi-day SWING / momentum priority (holds of 1-5 days), NOT intraday scalping
- Prefer liquid names with sustained upward momentum across several days
- Prefer moderate gaps and healthy volume over extreme one-day spikes that often fade
- Deprioritize exhaustion / gap-and-crap candidates and names with risky fresh news"""

REACT_WATCHLIST_PROMPT = """You rank a stock watchlist by trading priority. Work step by step: think, optionally call a
read-only tool to investigate a symbol, read the observation, and repeat. Then give the final ranking.
The lowest-ranked symbols will be blocked from new entries until the next ranking, so rank carefully.

Rules:
{rules}
- If data_quality is "partial", use research_price / research_change_pct; null live metrics alone are not a reason to rank last
- Investigate symbols where the table is ambiguous; don't waste calls on obvious cases
- Rank EVERY symbol in the table, best first

Tools:
{tools}

Watchlist metrics:
{seed}

Steps so far:
{transcript}

{budget}

Respond with ONE JSON object only — no markdown, no prose. Keep "thought" to one short sentence.
Either a tool call:
{{"thought": "why", "action": "<tool name>", "args": {{"symbol": "XYZ"}}}}
or the final answer:
{{"thought": "why", "action": "final", "args": {{"ranked": ["SYMBOL1", "SYMBOL2", ...], "reason": "brief reason"}}}}"""

ALERT_SUMMARY_PROMPT = """Summarize this trading session for the user in 2-3 sentences:
{context}"""

PREMARKET_BRIEFING_PROMPT = """You are a pre-market trading risk analyst. Review overnight news and flag symbols to avoid today.
Respond with JSON only, no other text.

Watchlist: {symbols}

News headlines (last 24h):
{news}

Keyword flags (earnings, FDA, guidance, downgrade, lawsuit):
{keyword_flags}

Rules:
- Put symbols with earnings today, major negative news, or high event risk in "avoid"
- Put symbols with mixed but manageable news in "caution"
- Only include symbols from the watchlist

Respond exactly:
{{"avoid": ["SYMBOL"], "caution": ["SYMBOL"], "reason": "brief summary"}}"""

SCREENER_RANK_PROMPT = """You are a scalping stock screener. Pick the best symbols for intraday scalping today.
Respond with JSON only, no other text.

Pick exactly {slots} symbols from the candidates below.
Prefer: high volume, moderate gap (0.5-3%), liquid large-caps, clear catalyst in headline.
Avoid: extreme gaps (>5%), low volume, earnings risk.

Candidates:
{candidates}

Respond exactly:
{{"picks": ["SYM1", "SYM2", "SYM3"], "reasons": {{"SYM1": "brief reason"}}, "summary": "one line"}}"""

SWING_SCREENER_RANK_PROMPT = """You are a swing/momentum stock screener. Pick symbols suitable for multi-day holds (1–5 days).
Respond with JSON only, no other text.

Pick exactly {slots} symbols from the candidates below.
Prefer: liquid names, clear catalyst, moderate gap (roughly 0.3–5%), volume confirmation, trend continuation.
Avoid: extreme gaps (>8%), illiquid names, pure one-day spike fades with no hold thesis, earnings landmines.

Candidates:
{candidates}

Respond exactly:
{{"picks": ["SYM1", "SYM2", "SYM3"], "reasons": {{"SYM1": "brief reason"}}, "summary": "one line"}}"""

SWING_REVIEW_PROMPT = """You are an intelligent swing trade position reviewer. A position has been held for {days_held} day(s).
Your job: decide whether to hold for more upside, exit now to lock profit/cut loss, or tighten the stop to protect gains.
Respond with JSON only, no other text.

This is a multi-day momentum swing — do NOT manage it like an intraday scalp. Mild early noise is normal.

Position context:
{context}

Rules:
- "hold": strong momentum, trend intact, catalyst still active — reasonable chance of reaching {take_profit_pct}%+ target
- "exit": trend clearly broken, catalyst faded, or max hold risk near — ONLY when you are confident
- "trail": momentum slowing but still positive — tighten stop to {trail_stop_pct}% below current high to lock profit
- NEVER recommend hold if pnl_pct <= -{hard_stop_pct}% (hard stop floor)
- If days_held >= {max_hold_days} - 1, prefer "exit" unless strong momentum
- For profitable trades: prefer "trail" over "hold" once pnl_pct > 1%
- If days_held is 0 or 1 and pnl is only mildly negative (above hard stop), prefer "hold" unless
  momentum is clearly broken (price below VWAP with bearish EMA and fading volume)
- Use outcome_cards when present: if similar recent trades died on same-day dynamic_stop after
  weak follow-through, be more willing to exit or trail; if similar trades worked via trailing_stop,
  prefer hold/trail
- Set confidence >= 0.7 only when the action is clear. If unsure, choose "hold" with lower confidence
  rather than "exit" — the system will ignore low-confidence exits.

Respond exactly:
{{"action": "hold" | "exit" | "trail", "confidence": 0.0-1.0, "new_stop_pct": null_or_float, "reason": "brief reason"}}
(new_stop_pct: tighter stop distance from current high in %, only set when action="trail")"""

EXIT_ADVISOR_PROMPT = """You are an intraday scalping exit advisor. A position is open; decide whether to sell now or hold for a target.
Respond with JSON only, no other text.

Zone: {zone}
(hard stop loss at -{hard_stop_pct}% is enforced by the system — you cannot widen it)

Position context:
{context}

Rules for zone "profit":
- "hold" only if momentum supports reaching target_pct within max_hold_minutes
- target_pct must be between {min_take_profit_pct} and {max_target_pct} (percent from entry)
- max_hold_minutes must be <= {max_hold_minutes}
- If unsure or session time is short, choose "sell"

Rules for zone "loss":
- "hold" only if a bounce to target_pct (>= 0, recovery toward breakeven/small profit) is plausible soon
- max_hold_minutes must be <= {max_loss_hold_minutes}
- Never recommend holding through the hard stop
- If unsure, choose "sell"

Respond exactly:
{{"action": "hold" or "sell", "target_pct": 0.0, "max_hold_minutes": 10, "confidence": 0.0-1.0, "reason": "brief reason"}}
For "sell", target_pct and max_hold_minutes may be null."""
