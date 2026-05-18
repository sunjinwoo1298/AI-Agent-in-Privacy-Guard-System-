"""Results aggregation and lightweight statistics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import accuracy_score, f1_score

from mas_privacy_eval.metrics.core import bootstrap_ci


AGG_COLS = [
    "f1",
    "precision",
    "recall",
    "fpr",
    "fnr",
    "accuracy",
    "mean_latency_ms",
    "p95_latency_ms",
    "mean_tokens",
    "mean_input_tokens",
    "mean_context_chars",
    "disagreement_rate",
    "escalation_rate",
    "parse_failure_rate",
    "parse_retry_rate",
]


def build_summary_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    df_summary = df_metrics.groupby(["topology", "n_agents"])[AGG_COLS].agg(["mean", "std"]).reset_index()
    df_summary.columns = ["topology", "n_agents"] + [
        f"{col}_{stat}" for col in AGG_COLS for stat in ["mean", "std"]
    ]
    return df_summary


def build_bootstrap_ci_table(df_metrics: pd.DataFrame) -> pd.DataFrame:
    ci_records = []
    for (topo, n_agents), grp in df_metrics.groupby(["topology", "n_agents"]):
        f1_vals = grp["f1"].tolist()
        lat_vals = grp["mean_latency_ms"].tolist()
        ci_f1 = bootstrap_ci(f1_vals)
        ci_lat = bootstrap_ci(lat_vals)
        ci_records.append(
            {
                "topology": topo,
                "n_agents": int(n_agents),
                "f1_ci_lo": ci_f1[0],
                "f1_ci_hi": ci_f1[1],
                "lat_ci_lo": ci_lat[0],
                "lat_ci_hi": ci_lat[1],
            }
        )
    return pd.DataFrame(ci_records)


def write_stats_report(df_metrics: pd.DataFrame, df_raw: pd.DataFrame, output_path: Path) -> None:
    """Write a compact stats report similar to the notebook output."""

    lines: List[str] = []

    if df_metrics.empty:
        output_path.write_text("No metrics available.\n", encoding="utf-8")
        return

    df_summary = build_summary_table(df_metrics)

    lines.append("Optimal agent count per topology (by F1/latency efficiency):")
    lines.append("─────────────────────────────────────────────────────────────────")
    for topo in sorted(df_summary["topology"].unique().tolist()):
        sub = df_summary[df_summary["topology"] == topo].copy()
        sub["eff"] = sub["f1_mean"] / (sub["mean_latency_ms_mean"].clip(lower=1.0))
        best = sub.loc[sub["eff"].idxmax()]
        lines.append(
            f"  {topo:<14}: N*={int(best['n_agents'])}  F1={best['f1_mean']:.3f}  Latency={best['mean_latency_ms_mean']:.0f}ms"
        )

    lines.append("")
    lines.append("Full summary table (mean ± std):")
    display_cols = [
        "topology",
        "n_agents",
        "f1_mean",
        "f1_std",
        "mean_latency_ms_mean",
        "mean_latency_ms_std",
        "mean_tokens_mean",
        "disagreement_rate_mean",
    ]
    lines.append(df_summary[display_cols].round(3).to_string(index=False))

    # Mann-Whitney U across topologies for F1
    lines.append("")
    lines.append("Mann-Whitney U tests between topologies (F1):")
    topos = df_metrics["topology"].unique().tolist()
    for i in range(len(topos)):
        for j in range(i + 1, len(topos)):
            g1 = df_metrics[df_metrics["topology"] == topos[i]]["f1"].tolist()
            g2 = df_metrics[df_metrics["topology"] == topos[j]]["f1"].tolist()
            if len(g1) >= 2 and len(g2) >= 2:
                u, p = mannwhitneyu(g1, g2, alternative="two-sided")
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                lines.append(f"  {topos[i]:<14} vs {topos[j]:<14}: U={u:.0f}, p={p:.4f} {sig}")

    # Spearman: N vs F1 per topology
    lines.append("")
    lines.append("Spearman correlation: N_agents vs F1 per topology:")
    for topo in df_metrics["topology"].unique():
        sub = df_metrics[df_metrics["topology"] == topo]
        if len(sub) >= 4:
            r, p = spearmanr(sub["n_agents"], sub["f1"])
            lines.append(f"  {topo:<14}: ρ={r:.3f}, p={p:.4f}")

    # Performance by category
    lines.append("")
    lines.append("Detection performance by sample category:")
    if not df_raw.empty and "category" in df_raw.columns and "pred" in df_raw.columns:
        df_valid = df_raw.dropna(subset=["pred"])
        for cat in sorted(df_valid["category"].unique().tolist()):
            sub = df_valid[df_valid["category"] == cat]
            if sub.empty or sub["true_label"].nunique() < 2:
                continue
            y_true = sub["true_label"].astype(int)
            y_pred = sub["pred"].astype(int)
            f1_c = f1_score(y_true, y_pred, zero_division=0)
            acc_c = accuracy_score(y_true, y_pred)
            lines.append(f"  {cat:<20}: n={len(sub)}, F1={f1_c:.3f}, Acc={acc_c:.3f}")

    # Latency scaling analysis
    lines.append("")
    lines.append("Latency scaling analysis:")
    for topo in df_summary["topology"].unique():
        sub = df_summary[df_summary["topology"] == topo].sort_values("n_agents")
        if len(sub) >= 3:
            x = sub["n_agents"].values.astype(float)
            y = sub["mean_latency_ms_mean"].values.astype(float)
            lin_coeff = np.polyfit(x, y, 1)
            quad_coeff = np.polyfit(x, y, 2)
            lin_r2 = 1 - np.sum((y - np.polyval(lin_coeff, x)) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-9)
            quad_r2 = 1 - np.sum((y - np.polyval(quad_coeff, x)) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-9)
            model = "quadratic" if quad_r2 > lin_r2 + 0.05 else "linear"
            lines.append(f"  {topo:<14}: best fit={model} (lin R²={lin_r2:.3f}, quad R²={quad_r2:.3f})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
