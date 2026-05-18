"""Sampling utilities for experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from mas_privacy_eval.data.models import PrivacySample


def stratified_sample_by_difficulty(samples: List[PrivacySample], n: int, seed: int) -> List[PrivacySample]:
    """Sample approximately evenly across difficulty tiers."""

    rng = np.random.default_rng(seed)
    by_diff: Dict[str, List[PrivacySample]] = defaultdict(list)
    for s in samples:
        by_diff[s.difficulty].append(s)

    diffs = sorted(by_diff.keys())
    if not diffs:
        return []

    per = n // len(diffs)
    remainder = n % len(diffs)

    picked: List[PrivacySample] = []
    for i, d in enumerate(diffs):
        k = per + (1 if i < remainder else 0)
        pool = by_diff[d]
        if not pool:
            continue

        if k <= len(pool):
            idx = rng.choice(len(pool), size=k, replace=False)
        else:
            idx = rng.choice(len(pool), size=k, replace=True)
        picked.extend([pool[j] for j in idx])

    rng.shuffle(picked)
    return picked[:n]
