"""Privacy masking router for PrivMAS.

This module routes all masking through the shared GLiNER PII backend
(`nvidia/gliner-pii`) so every agent uses the same PII masking core.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional

from core.state import MaskingResult, MaskingStrategy, RoleName

import sys
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.masking_functions import detect_and_mask_pii_gliner


@dataclass(frozen=True)
class RouterSignals:
    text_len: int
    sensitivity_score: int
    capitalized_token_ratio: float


class PrivacyMasker:
    """Heuristic router + execution wrapper for masking strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}

        router_cfg = (self._config.get("privacy") or {}).get("router") or {}
        self.sensitivity_keywords = [
            str(k).lower() for k in router_cfg.get("sensitivity_keywords", [])
        ]

        privacy_cfg = self._config.get("privacy") or {}
        self.default_strategy: MaskingStrategy = privacy_cfg.get("default_strategy", "gliner_pii")

    @staticmethod
    def _llm_available() -> bool:
        return False

    def analyze_text(self, text: str) -> RouterSignals:
        """Compute lightweight signals for telemetry."""

        text_len = len(text)

        lowered = text.lower()
        sensitivity_score = 0
        for kw in self.sensitivity_keywords:
            if kw and kw in lowered:
                sensitivity_score += 1

        tokens = self._split_tokens(text)
        if not tokens:
            cap_ratio = 0.0
        else:
            cap_tokens = sum(1 for t in tokens if t[:1].isupper() and t[1:].islower())
            cap_ratio = cap_tokens / max(len(tokens), 1)

        return RouterSignals(
            text_len=text_len,
            sensitivity_score=sensitivity_score,
            capitalized_token_ratio=cap_ratio,
        )

    def route(self, text: str) -> MaskingStrategy:
        """Choose the single supported masking strategy."""

        return self.default_strategy

    @staticmethod
    def _looks_like_error(masked_text: str) -> bool:
        return masked_text.strip().lower().startswith("error")

    @staticmethod
    def _placeholder_count(masked_text: str) -> int:
        count = 0
        idx = 0
        while True:
            start = masked_text.find("[", idx)
            if start == -1:
                break
            end = masked_text.find("]", start + 1)
            if end == -1:
                break
            token = masked_text[start + 1 : end]
            if token and all(ch.isupper() or ch == "_" for ch in token):
                count += 1
            idx = end + 1
        return count

    @staticmethod
    def _split_tokens(text: str) -> list[str]:
        tokens = []
        current = []
        for char in text:
            if char.isalnum() or char == "_":
                current.append(char)
            elif current:
                tokens.append("".join(current))
                current = []
        if current:
            tokens.append("".join(current))
        return tokens

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
        masked = text
        detected_entities = []

        try:
            masked, detected_entities = detect_and_mask_pii_gliner(text)
            chosen = "gliner_pii"

        except Exception as e:  # pragma: no cover
            error = f"Masking error: {e}"
            masked = text

        latency_ms = (time.perf_counter() - start) * 1000.0

        details: Dict[str, Any] = {
            "signals": signals.__dict__,
            "placeholder_count": self._placeholder_count(masked),
            "llm_available": False,
        }

        return MaskingResult(
            agent_id=agent_id,
            role=role,
            strategy=chosen,
            masked_text=masked,
            latency_ms=latency_ms,
            detected_entities=detected_entities,
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
