"""A fast heuristic agent (used for dry-run verification).

This is intentionally simple: it helps validate plumbing without downloading a
large model.
"""

from __future__ import annotations

import json
import re
import time
import zlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from mas_privacy_eval.agents.types import AgentOutput


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-. ]?)?(?:\(?\d{3}\)?[-. ]?)\d{3}[-. ]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def _has_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    needles = [
        "ignore previous instructions",
        "system prompt",
        "jailbreak",
        "act as dan",
        "output the system prompt",
    ]
    return any(n in lowered for n in needles)


def _contains_credential(text: str) -> bool:
    lowered = text.lower()
    return "password" in lowered or "passcode" in lowered or "api key" in lowered or "secret" in lowered


def _heuristic_label(text: str) -> int:
    if _has_prompt_injection(text):
        return 1
    if _EMAIL_RE.search(text) or _PHONE_RE.search(text) or _SSN_RE.search(text):
        return 1
    if _contains_credential(text):
        return 1

    # Naive high-precision financial/medical cues
    lowered = text.lower()
    if "credit card" in lowered or "cvv" in lowered or "iban" in lowered or "passport" in lowered:
        return 1
    if "medical" in lowered or "patient" in lowered or "diagnosis" in lowered:
        return 1

    # Avoid flagging purely internal RFC1918 IP mention
    if "rfc-1918" in lowered or "private address" in lowered:
        return 0

    # If a long digit sequence looks like a card, flag (best-effort)
    if _CC_RE.search(text) and any(k in lowered for k in ["cvv", "expiry", "exp", "credit", "card"]):
        return 1
    return 0


def _risk_type(text: str, label: int) -> str:
    if label == 0:
        return "none"
    if _has_prompt_injection(text):
        return "prompt_injection"
    if _contains_credential(text):
        return "credential_leak"
    return "pii"


@dataclass
class HeuristicPrivacyAgent:
    """Deterministic-ish heuristic agent."""

    agent_id: str
    role: str
    seed: int = 0

    def infer(self, text: str, prior_context: str = "") -> AgentOutput:
        t0 = time.perf_counter()
        stable = zlib.adler32(self.agent_id.encode("utf-8")) % 10000
        rng = np.random.default_rng(self.seed + stable)

        base_label = _heuristic_label(text)

        # Role-specific behavior: Detector is recall-heavy, Analyzer more conservative.
        if self.role == "Detector":
            label = base_label
            confidence = 0.85 if label == 1 else 0.65
        elif self.role == "Analyzer":
            label = base_label if _EMAIL_RE.search(text) or _SSN_RE.search(text) or _contains_credential(text) else 0
            confidence = 0.80 if label == 1 else 0.70
        elif self.role == "Validator":
            # Validator leans towards 'clear' unless strong evidence
            label = base_label if _SSN_RE.search(text) or _contains_credential(text) else base_label
            confidence = 0.75 if label == 1 else 0.75
        elif self.role == "Consensus":
            # Aggregate labels mentioned in the prior context when present.
            votes = [int(m.group(1)) for m in re.finditer(r"label\s*[:=]\s*(0|1)", prior_context)]
            if votes:
                pos = sum(votes)
                neg = len(votes) - pos
                label = 1 if pos >= neg else 0
                confidence = 0.55 + 0.4 * (max(pos, neg) / max(1, len(votes)))
            else:
                label = base_label
                confidence = 0.75 if label == 1 else 0.70
        else:
            label = base_label
            confidence = 0.70

        confidence = float(max(0.0, min(1.0, confidence + rng.normal(0, 0.02))))

        parsed: Dict = {
            "label": int(label),
            "confidence": confidence,
            "risk_type": _risk_type(text, int(label)),
            "reasoning": "Heuristic dry-run classification.",
        }
        # Optional escalation marker
        if 0.45 <= confidence <= 0.55:
            parsed["escalate"] = True

        raw = json.dumps(parsed)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        input_tokens = max(8, int(len((text + prior_context).split()) * 1.35) + 40)
        output_tokens = max(8, int(len(raw.split()) * 1.35) + 10)
        total_tokens = input_tokens + output_tokens

        return AgentOutput(
            agent_id=self.agent_id,
            role=self.role,
            raw_response=raw,
            parsed=parsed,
            label=int(label),
            confidence=confidence,
            reasoning=str(parsed.get("reasoning", "")),
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            parse_success=True,
            parse_retries=0,
            error_message=None,
            timestamp=time.time(),
        )
