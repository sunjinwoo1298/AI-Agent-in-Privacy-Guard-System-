"""Performance + systems metrics.

Ported from the original notebook with the same definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from mas_privacy_eval.pipelines.types import PipelineResult


@dataclass(frozen=True)
class ExperimentMetrics:
    # Identifiers
    topology: str
    n_agents: int
    trial: int
    seed: int

    # Performance
    f1: float
    precision: float
    recall: float
    fpr: float
    fnr: float
    accuracy: float
    parse_failure_rate: float

    # Systems
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_tokens: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_context_chars: float

    # Coordination
    disagreement_rate: float
    escalation_rate: float
    parse_retry_rate: float

    n_samples: int


def compute_metrics(
    results: List[PipelineResult], topology: str, n_agents: int, trial: int, seed: int
) -> ExperimentMetrics:
    """Compute all metrics from a list of PipelineResult objects."""

    valid = [r for r in results if r.final_prediction is not None]
    n = len(valid)
    if n == 0:
        return ExperimentMetrics(
            topology=topology,
            n_agents=n_agents,
            trial=trial,
            seed=seed,
            f1=0.0,
            precision=0.0,
            recall=0.0,
            fpr=0.0,
            fnr=0.0,
            accuracy=0.0,
            parse_failure_rate=1.0,
            mean_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            mean_tokens=0.0,
            mean_input_tokens=0.0,
            mean_output_tokens=0.0,
            mean_context_chars=0.0,
            disagreement_rate=0.0,
            escalation_rate=0.0,
            parse_retry_rate=0.0,
            n_samples=0,
        )

    y_true = [r.true_label for r in valid]
    y_pred = [int(r.final_prediction) for r in valid]

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    acc = float(sum(t == p for t, p in zip(y_true, y_pred)) / n)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, sum(y_true))
    fpr = float(fp / (fp + tn + 1e-9))
    fnr = float(fn / (fn + tp + 1e-9))

    latencies = [r.total_latency_ms for r in valid]
    tokens = [r.total_tokens for r in valid]
    in_tok = [r.total_input_tokens for r in valid]
    out_tok = [r.total_output_tokens for r in valid]
    ctx_chars = [r.context_chars for r in valid]

    parse_failure_rate = float(np.mean([r.parse_failures > 0 for r in results]))
    disagreement_rate = float(np.mean([r.disagreement for r in valid]))
    escalation_rate = float(np.mean([r.escalated for r in valid]))
    parse_retry_rate = float(np.mean([r.parse_retries for r in valid]))

    return ExperimentMetrics(
        topology=topology,
        n_agents=n_agents,
        trial=trial,
        seed=seed,
        f1=f1,
        precision=prec,
        recall=rec,
        fpr=fpr,
        fnr=fnr,
        accuracy=acc,
        parse_failure_rate=parse_failure_rate,
        mean_latency_ms=float(np.mean(latencies)),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        mean_tokens=float(np.mean(tokens)),
        mean_input_tokens=float(np.mean(in_tok)),
        mean_output_tokens=float(np.mean(out_tok)),
        mean_context_chars=float(np.mean(ctx_chars)),
        disagreement_rate=disagreement_rate,
        escalation_rate=escalation_rate,
        parse_retry_rate=parse_retry_rate,
        n_samples=n,
    )


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval over the mean."""

    if not values:
        return (0.0, 0.0)

    rng_b = np.random.default_rng(0)
    boot_means = [
        float(np.mean(rng_b.choice(values, size=len(values), replace=True))) for _ in range(int(n_bootstrap))
    ]
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.percentile(boot_means, 100 * (1 - alpha))),
    )
