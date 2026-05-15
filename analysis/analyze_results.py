#!/usr/bin/env python3
"""Analyze experiment results to extract numeric insights and recommendations.

Reads `results/summary.csv` and `results/logs.csv` and computes:
- coordination tax fraction
- throughput (tokens/sec)
- correlations and percent changes
- recommended `n` values based on F1/latency tradeoff

Saves a short textual report to `results/analysis.txt` and prints to stdout.
"""

import os
import pandas as pd


def human(ms):
    return f"{ms:.2f} ms"


def main():
    summary_path = "results/summary.csv"
    logs_path = "results/logs.csv"
    if not os.path.exists(summary_path) or not os.path.exists(logs_path):
        raise FileNotFoundError("Required results files not found. Run experiments first.")

    summary = pd.read_csv(summary_path)
    logs = pd.read_csv(logs_path)

    # Derived: coordination tax fraction
    summary = summary.sort_values("n_agents")
    summary["coord_frac"] = summary["coordination_tax_mean"] / summary["total_latency_mean"]

    # Throughput per run in logs
    logs["throughput"] = logs["tokens_total"] / logs["total_latency"]

    # (optional) aggregate throughput stats by n if needed
    # thr = logs.groupby("n_agents")["throughput"].agg(["mean", "std"]) 

    # Compose report
    lines = []
    lines.append("Experiment analysis report")
    lines.append("========================\n")

    lines.append("Summary per n_agents:")
    for _, r in summary.iterrows():
        lines.append(f"n={int(r['n_agents'])}: total_latency_mean={r['total_latency_mean']:.4f}s, total_latency_std={r['total_latency_std']:.4f}s, total_latency_p95={r['total_latency_p95']:.4f}s")
        lines.append(f"  coordination_tax_mean={r['coordination_tax_mean']:.4f}s ({r['coord_frac']*100:.1f}% of total), sync_delay_mean={r['sync_delay_mean']:.4f}s, f1_mean={r['f1_mean']:.4f}")

    # Percent change from n=1 to max n
    n1 = summary.iloc[0]
    nmax = summary.iloc[-1]
    pct_increase = (nmax['total_latency_mean'] - n1['total_latency_mean']) / n1['total_latency_mean'] * 100 if n1['total_latency_mean'] > 0 else float('nan')
    lines.append("")
    lines.append(f"Total latency increased by {pct_increase:.1f}% from n={int(n1['n_agents'])} to n={int(nmax['n_agents'])}.")

    # Coordination tax observation
    avg_coord_frac = summary['coord_frac'].mean()
    lines.append(f"Average coordination tax fraction across n: {avg_coord_frac*100:.1f}%")
    if avg_coord_frac > 0.5:
        lines.append("Interpretation: coordination overhead dominates compute; adding agents mainly increases waiting/merge overhead for this workload.")
    else:
        lines.append("Interpretation: compute and coordination are comparable; parallelism may provide benefits for larger workloads.")

    # Throughput insights
    tstats = logs.groupby('n_agents')['throughput'].agg(['mean','std'])
    lines.append("")
    lines.append("Throughput (tokens/sec) mean by n_agents:")
    for n, row in tstats.iterrows():
        lines.append(f"  n={int(n)}: mean={row['mean']:.1f} tok/s, std={row['std']:.1f}")

    # Accuracy stability
    f1_by_n = logs.groupby('n_agents')['f1'].agg(['mean','std'])
    lines.append("")
    lines.append("Accuracy (F1) by n_agents:")
    for n, row in f1_by_n.iterrows():
        lines.append(f"  n={int(n)}: mean_f1={row['mean']:.3f}, std={row['std']:.3f}")

    # Best tradeoff metric: prefer T_inf/L_sys (efficiency) when available, else fallback to F1/latency
    lines.append("")
    if 'efficiency_mean' in summary.columns:
        best = summary[['n_agents', 'efficiency_mean']].sort_values('efficiency_mean', ascending=False).iloc[0]
        lines.append(f"Best T_inf/L_sys tradeoff at n={int(best['n_agents'])} (efficiency_mean={best['efficiency_mean']:.3f})")
    else:
        candidate = summary[['n_agents','total_latency_mean','f1_mean']].copy()
        candidate['efficiency'] = candidate['f1_mean'] / candidate['total_latency_mean']
        best = candidate.sort_values('efficiency', ascending=False).iloc[0]
        lines.append(f"Best F1/latency tradeoff at n={int(best['n_agents'])} (efficiency={best['efficiency']:.3f})")

    out = "\n".join(lines)
    print(out)

    # save
    with open('results/analysis.txt', 'w', encoding='utf-8') as f:
        f.write(out)
    print('\nWrote results/analysis.txt')


if __name__ == '__main__':
    main()
