"""Evaluation metrics for PrivMAS.

These metrics are designed for *ablation studies* where you vary:
- number of specialists per role (detector/analyzer/validator)
- aggregation policy

We focus on measurements you can compute reliably from the current code:
- latency (per agent + end-to-end)
- coordination overhead proxies (message count + bytes)
- residual PII heuristics (emails/phones still present)
- placeholder statistics (how much text was masked)

Token usage
-----------
The existing `detect_and_mask_pii_llm` returns only a string, not provider usage.
So we provide an *approximate* token estimator for comparability.
"""

from __future__ import annotations

from typing import Any, Dict

from core.state import PrivMASState
from evaluation.token_tracking import sum_usage


def approx_token_count(text: str) -> int:
    """Very rough token approximation (provider-agnostic).

    This is NOT a billing-accurate tokenizer. It's meant for relative comparisons
    across agent counts / strategies.
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


def _normalise_tokens(text: str) -> list[str]:
    tokens = []
    for raw in text.split():
        token = raw.strip(".,;:!?()[]{}<>\"'")
        if token:
            tokens.append(token)
    return tokens


def _looks_like_email(token: str) -> bool:
    if token.count("@") != 1:
        return False
    local, domain = token.split("@", 1)
    return bool(local and domain and "." in domain and " " not in token)


def _looks_like_phone(token: str) -> bool:
    digits = [ch for ch in token if ch.isdigit()]
    if len(digits) < 10 or len(digits) > 15:
        return False
    return any(ch in token for ch in "-. ()") or token.isdigit()


def _count_placeholders(text: str) -> int:
    count = 0
    idx = 0
    while True:
        start = text.find("[", idx)
        if start == -1:
            break
        end = text.find("]", start + 1)
        if end == -1:
            break
        token = text[start + 1 : end]
        if token and all(ch.isupper() or ch == "_" for ch in token):
            count += 1
        idx = end + 1
    return count


def privacy_audit(masked_text: str) -> Dict[str, Any]:
    """Heuristic audit: checks for obvious residual PII patterns."""

    tokens = _normalise_tokens(masked_text)
    emails_left = [token for token in tokens if _looks_like_email(token)]
    phones_left = [token for token in tokens if _looks_like_phone(token)]

    return {
        "residual_email_count": len(emails_left),
        "residual_phone_count": len(phones_left),
        "residual_emails": emails_left[:3],
        "residual_phones": phones_left[:3],
        "placeholder_count": _count_placeholders(masked_text),
    }


def coordination_overhead(state: PrivMASState) -> Dict[str, Any]:
    """Compute coordination overhead proxies from message logs."""

    msg_count = len(state.messages)
    bytes_total = sum(m.content_bytes for m in state.messages)

    # Approximate tokens moved in coordination messages
    approx_tokens_total = sum(approx_token_count(m.content) for m in state.messages)

    return {
        "message_count": msg_count,
        "message_bytes": bytes_total,
        "message_tokens_approx": approx_tokens_total,
    }


def latency_metrics(state: PrivMASState) -> Dict[str, Any]:
    """Summarize latencies from state.timings_ms and per-result latency."""

    per_agent = {}
    for agent_id, result in state.specialist_results.items():
        per_agent[agent_id] = result.latency_ms

    e2e_ms = state.timings_ms.get("e2e_ms")

    return {
        "e2e_ms": e2e_ms,
        "per_agent_ms": per_agent,
        "sum_specialists_ms": sum(per_agent.values()),
    }


def compute_all_metrics(state: PrivMASState) -> Dict[str, Any]:
    """Compute and attach all metrics for a completed run."""

    masked = state.final_masked_text or ""

    # --- LLM token usage ---
    specialist_exact = []
    specialist_approx = []
    for r in state.specialist_results.values():
        details = r.details or {}
        specialist_exact.append(details.get("llm_usage"))
        specialist_approx.append(details.get("llm_usage_approx"))

    supervisor_usage = (state.metrics or {}).get("llm_usage", {})
    supervisor_exact = [supervisor_usage.get("supervisor_plan")]
    supervisor_approx = [supervisor_usage.get("supervisor_plan_approx")]

    llm_tokens = {
        "specialists_exact": sum_usage(specialist_exact),
        "specialists_approx": sum_usage(specialist_approx),
        "supervisor_plan_exact": sum_usage(supervisor_exact),
        "supervisor_plan_approx": sum_usage(supervisor_approx),
    }

    llm_tokens["total_exact"] = {
        "calls": llm_tokens["specialists_exact"]["calls"] + llm_tokens["supervisor_plan_exact"]["calls"],
        "prompt_tokens": llm_tokens["specialists_exact"]["prompt_tokens"] + llm_tokens["supervisor_plan_exact"]["prompt_tokens"],
        "completion_tokens": llm_tokens["specialists_exact"]["completion_tokens"] + llm_tokens["supervisor_plan_exact"]["completion_tokens"],
        "total_tokens": llm_tokens["specialists_exact"]["total_tokens"] + llm_tokens["supervisor_plan_exact"]["total_tokens"],
    }
    llm_tokens["total_approx"] = {
        "calls": llm_tokens["specialists_approx"]["calls"] + llm_tokens["supervisor_plan_approx"]["calls"],
        "prompt_tokens": llm_tokens["specialists_approx"]["prompt_tokens"] + llm_tokens["supervisor_plan_approx"]["prompt_tokens"],
        "completion_tokens": llm_tokens["specialists_approx"]["completion_tokens"] + llm_tokens["supervisor_plan_approx"]["completion_tokens"],
        "total_tokens": llm_tokens["specialists_approx"]["total_tokens"] + llm_tokens["supervisor_plan_approx"]["total_tokens"],
    }

    return {
        "latency": latency_metrics(state),
        "coordination": coordination_overhead(state),
        "privacy_audit": privacy_audit(masked),
        "llm_tokens": llm_tokens,
        "input_tokens_approx": approx_token_count(state.text),
        "output_tokens_approx": approx_token_count(masked),
    }
