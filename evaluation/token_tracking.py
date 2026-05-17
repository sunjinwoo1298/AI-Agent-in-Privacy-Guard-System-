"""LLM token tracking utilities.

This project uses Groq via the `groq` Python SDK in `src/single_agent.py`.
That code intentionally returns only the masked text.

To support coordination-overhead studies, we provide:
- Exact token usage capture when the provider returns `response.usage`
- Approximate token estimates (provider-agnostic) as a fallback

Important
---------
We do NOT modify the existing masking functions in `src/single_agent.py`.
Instead, when we need exact usage we can *temporarily* wrap the Groq SDK
`chat.completions.create` method to capture the returned usage.

This is intentionally lightweight and scoped (context-manager) to keep the
system simple for experiments.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Dict, List, Optional


def approx_token_count(text: str) -> int:
    """Very rough token approximation.

    This is NOT billing-accurate. It is stable across environments and useful
    for relative comparisons.
    """

    count = 0
    current = []

    for char in text:
        if char.isalnum() or char == "_":
            current.append(char)
            continue

        if current:
            count += 1
            current = []

        if not char.isspace():
            count += 1

    if current:
        count += 1

    return count


def _normalize_usage(usage: Any) -> Optional[Dict[str, Any]]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return dict(usage)

    # Pydantic v2 style
    if hasattr(usage, "model_dump"):
        try:
            data = usage.model_dump()
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Pydantic v1 style
    if hasattr(usage, "dict"):
        try:
            data = usage.dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Plain object with attributes
    out: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if hasattr(usage, key):
            out[key] = getattr(usage, key)
    return out or None


# Public alias for other modules (Supervisor, etc.)
normalize_usage = _normalize_usage


class GroqTokenTracker(AbstractContextManager):
    """Context manager that captures Groq token usage for a single call.

    Usage
    -----
    >>> with GroqTokenTracker() as t:
    ...     text = detect_and_mask_pii_llm(input_text)
    >>> t.last_usage

    Notes
    -----
    - If Groq SDK is not installed or internal structure changes, this tracker
      becomes a no-op (enabled=False).
    - This is sufficient for this repo because LLM calls are sequential.
    """

    enabled: bool
    last_usage: Optional[Dict[str, Any]]
    last_usage_approx: Optional[Dict[str, Any]]
    calls_attempted: int
    calls_succeeded: int

    def __init__(self) -> None:
        self.enabled = False
        self.last_usage = None
        self.last_usage_approx = None
        self.calls_attempted = 0
        self.calls_succeeded = 0
        self._patched_cls = None
        self._orig_create = None

    def __enter__(self):
        try:
            from groq import Groq  # type: ignore
        except Exception:
            return self

        try:
            dummy = Groq(api_key="DUMMY")
            completions_obj = dummy.chat.completions
            completions_cls = type(completions_obj)
            orig_create = completions_cls.create

            def wrapped_create(self_obj, *args, **kwargs):  # type: ignore[no-redef]
                # Approximate from the request/response payload if available
                prompt_text = ""
                messages = kwargs.get("messages")
                if isinstance(messages, list) and messages:
                    # Common pattern: first user message contains the full prompt
                    first = messages[0] if isinstance(messages[0], dict) else None
                    if first and isinstance(first.get("content"), str):
                        prompt_text = first["content"]

                self.calls_attempted += 1

                try:
                    resp = orig_create(self_obj, *args, **kwargs)
                    self.calls_succeeded += 1

                    usage = _normalize_usage(getattr(resp, "usage", None))
                    self.last_usage = usage

                    completion_text = ""
                    try:
                        completion_text = resp.choices[0].message.content or ""
                    except Exception:
                        completion_text = ""

                    self.last_usage_approx = {
                        "prompt_tokens": approx_token_count(prompt_text),
                        "completion_tokens": approx_token_count(completion_text),
                        "total_tokens": approx_token_count(prompt_text)
                        + approx_token_count(completion_text),
                        "model": kwargs.get("model"),
                    }

                    return resp

                except Exception:
                    # Even if the call errors, we can still record approximate prompt tokens.
                    self.last_usage = None
                    self.last_usage_approx = {
                        "prompt_tokens": approx_token_count(prompt_text),
                        "completion_tokens": 0,
                        "total_tokens": approx_token_count(prompt_text),
                        "model": kwargs.get("model"),
                    }
                    raise

            completions_cls.create = wrapped_create  # type: ignore[assignment]
            self._patched_cls = completions_cls
            self._orig_create = orig_create
            self.enabled = True
            return self

        except Exception:
            # No-op on any unexpected SDK structure
            return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self._patched_cls is not None and self._orig_create is not None:
            try:
                self._patched_cls.create = self._orig_create  # type: ignore[assignment]
            except Exception:
                pass
        return False


def sum_usage(usages: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Sum usage dicts that contain prompt/completion/total tokens."""

    prompt = 0
    completion = 0
    total = 0
    calls = 0

    for u in usages:
        if not u or not isinstance(u, dict):
            continue
        calls += 1
        prompt += int(u.get("prompt_tokens") or 0)
        completion += int(u.get("completion_tokens") or 0)
        total += int(u.get("total_tokens") or 0)

    return {
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
