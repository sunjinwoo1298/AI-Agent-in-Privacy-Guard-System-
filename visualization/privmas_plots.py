#!/usr/bin/env python3
"""PrivMAS plotting utilities.

This file is intentionally separate from `visualization/plots.py` because that module
is focused on the deterministic Thread/ProcessPool experiments.

Primary plot for the research question:
- Privacy quality vs coordination overhead across agent counts.

Inputs:
- CSV produced by `experiments/run_privmas_sweep.py`
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def plot_privacy_quality_vs_overhead(
    csv_path: str = "results/privmas_runs.csv",
    out_path: str = "results/plots/privmas/privacy_quality_vs_overhead.png",
    *,
    overhead_metric: str = "message_bytes",
    quality_metric: str = "residual_pii_count",
    title: Optional[str] = None,
) -> str:
    """Create a tradeoff plot: quality vs overhead.

    The sweep runner logs one row per (agent_config, run, sample). Here we aggregate
    by `n_per_role` (where detector/analyzer/validator counts are all equal) and plot
    a parametric curve in (overhead, residual_pii).

    Parameters
    ----------
    overhead_metric:
        One of: message_count, message_bytes, message_tokens_approx, e2e_ms
    quality_metric:
        One of: residual_pii_count, residual_email_count, residual_phone_count
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    required = {"n_per_role", overhead_metric, quality_metric}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Aggregate across samples + repeats for each agent count.
    g = (
        df.groupby("n_per_role")
        .agg(
            overhead_mean=(overhead_metric, "mean"),
            overhead_std=(overhead_metric, "std"),
            quality_mean=(quality_metric, "mean"),
            quality_std=(quality_metric, "std"),
            total_specialists_mean=("total_specialists", "mean"),
            rows=(quality_metric, "count"),
        )
        .reset_index()
        .sort_values("n_per_role")
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(g["overhead_mean"], g["quality_mean"], marker="o")

    # Annotate each point with the agent count (per role and total).
    for _, r in g.iterrows():
        n = int(r["n_per_role"])
        total = int(round(float(r["total_specialists_mean"])))
        plt.annotate(
            f"n={n} (tot={total})",
            (r["overhead_mean"], r["quality_mean"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    plot_title = title or f"PrivMAS tradeoff: {quality_metric} vs {overhead_metric}"
    plt.title(plot_title)
    plt.xlabel(f"Overhead ({overhead_metric}, mean)")
    plt.ylabel(f"Residual PII ({quality_metric}, mean; lower is better)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"Saved {out_path}")
    return out_path


__all__ = ["plot_privacy_quality_vs_overhead"]
