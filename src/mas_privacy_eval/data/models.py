"""Domain models for privacy evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacySample:
    """A single evaluation sample with ground truth and metadata."""

    sample_id: int
    text: str
    true_label: int  # 1 = privacy violation, 0 = clean
    token_count: int
    difficulty: str  # "easy" | "medium" | "hard"
    category: str  # "pii" | "prompt_injection" | "contextual" | "clean"
    source: str

    def approx_tokens(self) -> int:
        """Approximate token count from whitespace word count."""

        return max(4, int(len(self.text.split()) * 1.35))
