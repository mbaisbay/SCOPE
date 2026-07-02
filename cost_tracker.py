"""
cost_tracker.py
Per-outlet OpenAI usage and cost tracking.

Captures token counts from:
  - LangChain ChatOpenAI calls (via LangChainUsageCallback)
  - Native OpenAI Responses API (record_responses)
  - OpenAI embeddings API (record_embedding)

Each runner.run(item) is wrapped in a UsageTracker context. The tracker
aggregates tokens + web_search calls and converts them to USD using PRICING.
After the run, .snapshot() is appended to the .jsonl row, and a per-mode
summary file is written at the end of each combo.
"""

from __future__ import annotations

import contextvars
import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


# Pricing (May 2026) — verify against actual OpenAI invoice before reporting.
# Per-token rates (USD per 1 token, derived from $/1M).
PRICING: dict[str, dict[str, float]] = {
    "gpt-5-mini-2025-08-07": {
        "input": 0.25 / 1_000_000,
        "output": 2.00 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,  # placeholder: 10% of input
    },
    "text-embedding-3-small": {
        "input": 0.02 / 1_000_000,
    },
}

# Each web_search tool call is billed as a fixed input-token block.
WEB_SEARCH_BLOCK_TOKENS = 8000


def _price(model: str, kind: str) -> float:
    """Return per-token USD rate, falling back to 0.0 if unknown."""
    return PRICING.get(model, {}).get(kind, 0.0)


class UsageTracker:
    """Aggregates token usage across one outlet's evaluation run.

    Thread-safe — uses an internal lock so parallel runners (e.g., MBC's
    ThreadPoolExecutor) can update concurrently without races.
    """

    def __init__(self, model: str = "gpt-5-mini-2025-08-07", outlet_name: str = ""):
        self.model = model
        self.outlet_name = outlet_name
        self._lock = threading.Lock()
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.llm_cached_tokens = 0
        self.embedding_tokens = 0
        self.web_search_calls = 0
        self.calls: list[dict] = []

    def reset(self) -> None:
        with self._lock:
            self.llm_input_tokens = 0
            self.llm_output_tokens = 0
            self.llm_cached_tokens = 0
            self.embedding_tokens = 0
            self.web_search_calls = 0
            self.calls.clear()

    def record_chat(self, usage: dict, model: str | None = None) -> None:
        """Record token usage from a chat-completions call.

        `usage` is the dict shape from LangChain (`response.llm_output["token_usage"]`)
        or the OpenAI SDK (`response.usage.model_dump()`). Both expose
        prompt_tokens / completion_tokens; cached tokens may be nested under
        `prompt_tokens_details.cached_tokens` (SDK) or be absent (LangChain).
        """
        if not usage:
            return
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cached = 0
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens", 0) or 0)
        with self._lock:
            self.llm_input_tokens += prompt
            self.llm_output_tokens += completion
            self.llm_cached_tokens += cached
            self.calls.append({
                "kind": "chat",
                "model": model or self.model,
                "input": prompt, "output": completion, "cached": cached,
            })

    def record_responses(self, response: Any, model: str | None = None,
                         used_web_search: bool = False) -> None:
        """Record usage from a native OpenAI `responses.create()` call.

        If `used_web_search=True`, increments web_search_calls (each call is
        billed as one 8k-token input block in addition to whatever the
        response.usage already reports).
        """
        usage = getattr(response, "usage", None)
        usage_dict: dict = {}
        if usage is not None:
            try:
                usage_dict = usage.model_dump()
            except AttributeError:
                # Fall back to attribute access for older SDK shapes.
                usage_dict = {
                    "input_tokens": getattr(usage, "input_tokens", 0),
                    "output_tokens": getattr(usage, "output_tokens", 0),
                }
        prompt = int(usage_dict.get("input_tokens") or usage_dict.get("prompt_tokens") or 0)
        completion = int(usage_dict.get("output_tokens") or usage_dict.get("completion_tokens") or 0)
        cached = 0
        details = usage_dict.get("input_tokens_details") or usage_dict.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens", 0) or 0)
        with self._lock:
            self.llm_input_tokens += prompt
            self.llm_output_tokens += completion
            self.llm_cached_tokens += cached
            if used_web_search:
                self.web_search_calls += 1
            self.calls.append({
                "kind": "responses",
                "model": model or self.model,
                "input": prompt, "output": completion, "cached": cached,
                "web_search": used_web_search,
            })

    def record_embedding(self, response: Any, model: str = "text-embedding-3-small") -> None:
        """Record usage from an `embeddings.create()` call."""
        usage = getattr(response, "usage", None)
        tokens = 0
        if usage is not None:
            tokens = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0)
        with self._lock:
            self.embedding_tokens += tokens
            self.calls.append({
                "kind": "embedding",
                "model": model,
                "input": tokens,
            })

    def cost_usd(self) -> dict[str, float]:
        """Compute USD cost from current counters using PRICING."""
        # LLM: bill cached tokens at the cached rate, non-cached at full input rate.
        non_cached_input = max(self.llm_input_tokens - self.llm_cached_tokens, 0)
        llm_cost = (
            non_cached_input * _price(self.model, "input")
            + self.llm_cached_tokens * _price(self.model, "cached_input")
            + self.llm_output_tokens * _price(self.model, "output")
        )
        # web_search: 8k-token input block per call.
        web_cost = self.web_search_calls * WEB_SEARCH_BLOCK_TOKENS * _price(self.model, "input")
        embed_cost = self.embedding_tokens * _price("text-embedding-3-small", "input")
        return {
            "llm": round(llm_cost, 6),
            "web_search": round(web_cost, 6),
            "embedding": round(embed_cost, 6),
            "total": round(llm_cost + web_cost + embed_cost, 6),
        }

    def snapshot(self, include_calls: bool = False) -> dict:
        """Return a JSON-serializable summary of all usage so far."""
        with self._lock:
            snap = {
                "model": self.model,
                "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens,
                "llm_cached_tokens": self.llm_cached_tokens,
                "embedding_tokens": self.embedding_tokens,
                "web_search_calls": self.web_search_calls,
                "cost_usd": self.cost_usd(),
            }
            if include_calls:
                snap["calls"] = list(self.calls)
            return snap

    def add(self, snap: dict) -> None:
        """Fold another snapshot into this tracker (used for per-mode totals)."""
        with self._lock:
            self.llm_input_tokens += int(snap.get("llm_input_tokens", 0))
            self.llm_output_tokens += int(snap.get("llm_output_tokens", 0))
            self.llm_cached_tokens += int(snap.get("llm_cached_tokens", 0))
            self.embedding_tokens += int(snap.get("embedding_tokens", 0))
            self.web_search_calls += int(snap.get("web_search_calls", 0))


# ---------------------------------------------------------------------------
# Context plumbing — the "current" tracker for the outlet being processed
# ---------------------------------------------------------------------------

_current: contextvars.ContextVar[UsageTracker | None] = contextvars.ContextVar(
    "current_usage_tracker", default=None,
)


def current_tracker() -> UsageTracker | None:
    """Return the active UsageTracker for the calling context, or None."""
    return _current.get()


def set_current_tracker(tracker: UsageTracker | None):
    """Install `tracker` as the active one and return a token for reset.

    Usage:
        token = set_current_tracker(tracker)
        try:
            ...
        finally:
            reset_current_tracker(token)
    """
    return _current.set(tracker)


def reset_current_tracker(token) -> None:
    """Restore the previous tracker. Pair with set_current_tracker(...)."""
    _current.reset(token)


# ---------------------------------------------------------------------------
# LangChain callback — installed on ChatOpenAI so every invoke() is tracked
# ---------------------------------------------------------------------------

class LangChainUsageCallback(BaseCallbackHandler):
    """LangChain callback that forwards token usage to the current tracker.

    LangChain populates `response.llm_output['token_usage']` for ChatOpenAI
    after each call. We extract that and forward to the active UsageTracker
    (if any). When no tracker is active (e.g., metrics computation outside
    an outlet run), usage is silently dropped.
    """

    def on_llm_end(self, response, **kwargs) -> None:  # type: ignore[override]
        tracker = current_tracker()
        if tracker is None:
            return
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        model = llm_output.get("model_name") or llm_output.get("model")
        if usage:
            tracker.record_chat(usage, model=model)


def pricing_snapshot() -> dict:
    """Return a copy of the active PRICING + WEB_SEARCH_BLOCK_TOKENS."""
    return {
        "pricing": {m: dict(v) for m, v in PRICING.items()},
        "web_search_block_tokens": WEB_SEARCH_BLOCK_TOKENS,
    }
