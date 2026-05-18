"""Dataset loader.

Builds a multi-source privacy evaluation dataset from:
- a small curated corpus (always available)
- an optional Hugging Face dataset (best-effort)
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List

import numpy as np

from mas_privacy_eval.data.curated_corpus import CURATED_CORPUS
from mas_privacy_eval.data.models import PrivacySample


logger = logging.getLogger(__name__)


class PrivacyDatasetLoader:
    """Loads and preprocesses privacy evaluation samples."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def load_curated(self, n_samples: int, *, enable_augmentation: bool = True) -> List[PrivacySample]:
        """Load from curated corpus with light augmentation."""

        samples: List[PrivacySample] = []
        suffixes = ["", " [source: internal-audit]", " [tag: synthetic-variation]"]
        for i in range(n_samples):
            base = CURATED_CORPUS[i % len(CURATED_CORPUS)]
            text = base["text"]
            if enable_augmentation:
                text = text + self._rng.choice(suffixes)

            samples.append(
                PrivacySample(
                    sample_id=i,
                    text=text,
                    true_label=int(base["label"]),
                    token_count=max(4, int(len(text.split()) * 1.35)),
                    difficulty=str(base["diff"]),
                    category=str(base["cat"]),
                    source="curated",
                )
            )
        return samples

    def load_hf_ai4privacy(self, n_samples: int, *, max_text_chars: int = 500) -> List[PrivacySample]:
        """Best-effort load of ai4privacy/pii-masking-200k (streaming)."""

        try:
            from datasets import load_dataset

            ds = load_dataset(
                "ai4privacy/pii-masking-200k",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.warning("HuggingFace dataset unavailable (%s).", exc)
            return []

        samples: List[PrivacySample] = []
        idx = 0
        for row in ds:
            if idx >= n_samples:
                break

            text = (row.get("source_text") or "").strip()
            if len(text) < 20:
                continue

            has_pii = len(row.get("privacy_mask", []) or []) > 0
            text = text[:max_text_chars]

            samples.append(
                PrivacySample(
                    sample_id=10000 + idx,
                    text=text,
                    true_label=int(has_pii),
                    token_count=max(4, int(len(text.split()) * 1.35)),
                    difficulty="medium",
                    category="pii",
                    source="ai4privacy",
                )
            )
            idx += 1

        logger.info("Loaded %d samples from ai4privacy dataset", len(samples))
        return samples

    def load(
        self,
        *,
        n_curated: int,
        n_hf: int,
        max_hf_text_chars: int,
        enable_augmentation: bool,
    ) -> List[PrivacySample]:
        """Load full evaluation corpus."""

        samples = self.load_curated(n_curated, enable_augmentation=enable_augmentation)
        if n_hf > 0:
            hf_samples = self.load_hf_ai4privacy(n_hf, max_text_chars=max_hf_text_chars)
            offset = len(samples)
            for i, s in enumerate(hf_samples):
                hf_samples[i] = replace(s, sample_id=offset + i)
            samples.extend(hf_samples)
        return samples
