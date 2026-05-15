"""Privacy masking router for PrivMAS.

This module *reuses* the existing masking functions in `src/single_agent.py`:
- detect_and_mask_pii_regex
- detect_and_mask_pii_spacy
- detect_and_mask_pii_llm

Goal
----
Provide a small, configurable, rule-based router that chooses a masking strategy
based on quick heuristics (length, regex hits, sensitivity keywords, etc.).
This makes it easy to run ablation studies across different agent counts without
introducing heavy orchestration logic inside the masker itself.

Important
---------
We do NOT modify the existing functions in `src/single_agent.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Dict, Optional

from config import GROQ_API_KEY
from core.state import MaskingResult, MaskingStrategy, RoleName
from evaluation.token_tracking import GroqTokenTracker, approx_token_count

from src.single_agent import (
    detect_and_mask_pii_regex,
    detect_and_mask_pii_spacy,
    detect_and_mask_pii_llm,
)


_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_REGEX = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_PLACEHOLDER_REGEX = re.compile(r"\[[A-Z_]+\]")


@dataclass(frozen=True)
class RouterSignals:
    text_len: int
    regex_email_hits: int
    regex_phone_hits: int
    sensitivity_score: int
    capitalized_token_ratio: float

    @property
    def regex_hits(self) -> int:
        return self.regex_email_hits + self.regex_phone_hits


class PrivacyMasker:
    """Heuristic router + execution wrapper for masking strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}

        router_cfg = (self._config.get("privacy") or {}).get("router") or {}
        self.short_text_threshold: int = int(router_cfg.get("short_text_threshold", 200))
        self.long_text_threshold: int = int(router_cfg.get("long_text_threshold", 1200))
        self.sensitivity_keywords = [
            str(k).lower() for k in router_cfg.get("sensitivity_keywords", [])
        ]

        privacy_cfg = self._config.get("privacy") or {}
        self.default_strategy: MaskingStrategy = privacy_cfg.get(
            "default_strategy", "hybrid_fast"
        )
        self.llm_fallback_strategy: MaskingStrategy = privacy_cfg.get(
            "llm_fallback_strategy", "hybrid_fast"
        )

    @staticmethod
    def _llm_available() -> bool:
        return bool(GROQ_API_KEY) and GROQ_API_KEY != "YOUR_API_KEY"

    def analyze_text(self, text: str) -> RouterSignals:
        """Compute cheap signals for routing decisions."""

        text_len = len(text)
        email_hits = len(_EMAIL_REGEX.findall(text))
        phone_hits = len(_PHONE_REGEX.findall(text))

        lowered = text.lower()
        sensitivity_score = 0
        for kw in self.sensitivity_keywords:
            if kw and kw in lowered:
                sensitivity_score += 1

        tokens = re.findall(r"\b\w+\b", text)
        if not tokens:
            cap_ratio = 0.0
        else:
            cap_tokens = sum(1 for t in tokens if t[:1].isupper() and t[1:].islower())
            cap_ratio = cap_tokens / max(len(tokens), 1)

        return RouterSignals(
            text_len=text_len,
            regex_email_hits=email_hits,
            regex_phone_hits=phone_hits,
            sensitivity_score=sensitivity_score,
            capitalized_token_ratio=cap_ratio,
        )

    def route(self, text: str) -> MaskingStrategy:
        """Choose a masking strategy based on heuristics.

        Strategies (required by spec)
        ----------------------------
        - regex_only
        - spacy_plus
        - hybrid_fast
        - llm_only
        - hybrid_llm
        """

        s = self.analyze_text(text)

        # Very short, obvious-pattern texts → regex
        if s.text_len <= self.short_text_threshold and s.regex_hits > 0 and s.sensitivity_score == 0:
            return "regex_only"

        # High sensitivity keywords → prefer LLM hybrid if possible
        if s.sensitivity_score >= 2:
            return "hybrid_llm" if self._llm_available() else "hybrid_fast"

        # No regex hits but many proper-noun-like tokens → spaCy
        if s.regex_hits == 0 and s.capitalized_token_ratio >= 0.25:
            return "spacy_plus"

        # Very long inputs → avoid LLM by default (cost/latency)
        if s.text_len >= self.long_text_threshold:
            return "hybrid_fast"

        # Default middle-ground
        return self.default_strategy

    @staticmethod
    def _looks_like_error(masked_text: str) -> bool:
        return masked_text.strip().lower().startswith("error")

    @staticmethod
    def _placeholder_count(masked_text: str) -> int:
        return len(_PLACEHOLDER_REGEX.findall(masked_text))

    def mask(
        self,
        text: str,
        *,
        agent_id: str = "masker",
        role: RoleName = "detector",
        strategy: Optional[MaskingStrategy] = None,
    ) -> MaskingResult:
        """Mask text using a chosen or routed strategy."""

        chosen: MaskingStrategy = strategy or self.route(text)
        signals = self.analyze_text(text)

        start = time.perf_counter()
        error: Optional[str] = None
        llm_error: Optional[str] = None
        masked = text

        llm_usage: Optional[Dict[str, Any]] = None
        llm_usage_approx: Optional[Dict[str, Any]] = None

        try:
            if chosen == "regex_only":
                masked = detect_and_mask_pii_regex(text)

            elif chosen == "spacy_plus":
                masked = detect_and_mask_pii_spacy(text)

            elif chosen == "llm_only":
                prompt = self._build_llm_prompt(text)
                with GroqTokenTracker() as tracker:
                    masked = detect_and_mask_pii_llm(text)
                if self._looks_like_error(masked):
                    error = masked

                call_attempted = tracker.calls_attempted > 0 or (
                    self._llm_available()
                    and not masked.startswith("Error: Groq API key not configured")
                )
                if call_attempted:
                    llm_usage = tracker.last_usage
                    llm_usage_approx = tracker.last_usage_approx or {
                        "prompt_tokens": approx_token_count(prompt),
                        "completion_tokens": approx_token_count(masked),
                        "total_tokens": approx_token_count(prompt)
                        + approx_token_count(masked),
                        "model": "llama3-8b-8192",
                    }

            elif chosen == "hybrid_fast":
                masked = detect_and_mask_pii_spacy(detect_and_mask_pii_regex(text))

            elif chosen == "hybrid_llm":
                # Fast pre-mask, then ask LLM to catch harder PII.
                pre = detect_and_mask_pii_regex(text)
                prompt = self._build_llm_prompt(pre)
                with GroqTokenTracker() as tracker:
                    masked = detect_and_mask_pii_llm(pre)
                if self._looks_like_error(masked):
                    # Treat LLM failure as non-fatal if we can fall back to a
                    # deterministic strategy, so aggregation can still use the
                    # validator's final output.
                    llm_error = masked
                    chosen = self.llm_fallback_strategy
                    # Fallback keeps the pre-mask and adds spaCy if available.
                    masked = detect_and_mask_pii_spacy(pre)

                call_attempted = tracker.calls_attempted > 0 or (
                    self._llm_available()
                    and not masked.startswith("Error: Groq API key not configured")
                )
                if call_attempted:
                    llm_usage = tracker.last_usage
                    llm_usage_approx = tracker.last_usage_approx or {
                        "prompt_tokens": approx_token_count(prompt),
                        "completion_tokens": approx_token_count(masked),
                        "total_tokens": approx_token_count(prompt)
                        + approx_token_count(masked),
                        "model": "llama3-8b-8192",
                    }

            else:
                # Defensive fallback (should not happen if types are respected)
                masked = detect_and_mask_pii_regex(text)
                chosen = "regex_only"

        except Exception as e:  # pragma: no cover
            error = f"Masking error: {e}"
            masked = text

        latency_ms = (time.perf_counter() - start) * 1000.0

        details: Dict[str, Any] = {
            "signals": signals.__dict__,
            "placeholder_count": self._placeholder_count(masked),
            "llm_available": self._llm_available(),
        }

        if llm_usage is not None:
            details["llm_usage"] = llm_usage
        if llm_usage_approx is not None:
            details["llm_usage_approx"] = llm_usage_approx
        if llm_error is not None:
            details["llm_error"] = llm_error

        return MaskingResult(
            agent_id=agent_id,
            role=role,
            strategy=chosen,
            masked_text=masked,
            latency_ms=latency_ms,
            error=error,
            details=details,
        )

    @staticmethod
    def _build_llm_prompt(text: str) -> str:
        """Keep prompt construction aligned with `src/single_agent.py`.

        This is used only for approximate token counting.
        """

        return f"""
    Analyze the following text and identify any personally identifiable information (PII) such as names, emails, and phone numbers.
    Your task is to return the original text with the identified PII replaced by a corresponding placeholder (e.g., [NAME], [EMAIL], [PHONE]).
    Do not provide any explanation, only the masked text.

    Text: \"{text}\"
    """
