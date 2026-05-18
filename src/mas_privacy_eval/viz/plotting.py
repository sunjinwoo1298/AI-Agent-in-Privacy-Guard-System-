"""Save plots to disk (no notebook inline rendering)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "sequential": "#1f77b4",
    "parallel": "#ff7f0e",
    "hierarchical": "#2ca02c",
    "blackboard": "#d62728",
}

MARKERS = {"sequential": "o", "parallel": "s", "hierarchical": "D", "blackboard": "^"}


def save_results_plot(
    *,
    df_summary: pd.DataFrame,
    df_ci: pd.DataFrame,
    agent_counts: list[int],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_f1, ax_lat, ax_tok, ax_dis = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for topo in sorted(df_summary["topology"].unique().tolist()):
        sub = df_summary[df_summary["topology"] == topo].sort_values("n_agents")
        ci = df_ci[df_ci["topology"] == topo].sort_values("n_agents")
        color = COLORS.get(topo, "gray")
        marker = MARKERS.get(topo, "o")

        ax_f1.plot(sub["n_agents"], sub["f1_mean"], color=color, marker=marker, linewidth=2, label=topo)
        if not ci.empty:
            ax_f1.fill_between(ci["n_agents"], ci["f1_ci_lo"], ci["f1_ci_hi"], color=color, alpha=0.12)

        ax_lat.plot(
            sub["n_agents"],
            sub["mean_latency_ms_mean"],
            color=color,
            marker=marker,
            linewidth=2,
            label=topo,
        )
        if not ci.empty:
            ax_lat.fill_between(ci["n_agents"], ci["lat_ci_lo"], ci["lat_ci_hi"], color=color, alpha=0.12)

        ax_tok.plot(
            sub["n_agents"],
            sub["mean_tokens_mean"],
            color=color,
            marker=marker,
            linewidth=2,
            label=topo,
        )
        ax_dis.plot(
            sub["n_agents"],
            sub["disagreement_rate_mean"] * 100.0,
            color=color,
            marker=marker,
            linewidth=2,
            label=topo,
        )

    ax_f1.set_title("F1 vs Agent Count (shaded = 95% bootstrap CI)")
    ax_f1.set_xlabel("Number of Agents (N)")
    ax_f1.set_ylabel("F1")
    ax_f1.set_ylim(0.0, 1.05)
    ax_f1.set_xticks(agent_counts)
    ax_f1.legend(fontsize=9)

    ax_lat.set_title("Mean latency vs Agent Count")
    ax_lat.set_xlabel("Number of Agents (N)")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_xticks(agent_counts)

    ax_tok.set_title("Mean tokens vs Agent Count")
    ax_tok.set_xlabel("Number of Agents (N)")
    ax_tok.set_ylabel("Tokens")
    ax_tok.set_xticks(agent_counts)

    ax_dis.set_title("Disagreement rate vs Agent Count")
    ax_dis.set_xlabel("Number of Agents (N)")
    ax_dis.set_ylabel("Disagreement (%)")
    ax_dis.set_xticks(agent_counts)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
