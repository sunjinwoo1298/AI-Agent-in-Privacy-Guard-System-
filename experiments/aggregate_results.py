#!/usr/bin/env python3
"""Aggregate raw per-run CSV into summary statistics (mean/std/p95) per `n_agents`."""

import os
import pandas as pd


def aggregate(csv_path: str = "results/logs.csv", out_path: str = "results/summary.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    metrics = ["total_latency", "coordination_tax", "sync_delay", "precision", "recall", "f1", "efficiency", "total_throughput"]
    rows = []
    grouped = df.groupby("n_agents")
    for name, g in grouped:
        row = {"n_agents": int(name), "count": len(g)}
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std(ddof=0)
                row[f"{m}_p95"] = g[m].quantile(0.95)
            else:
                row[f"{m}_mean"] = None
                row[f"{m}_std"] = None
                row[f"{m}_p95"] = None
        rows.append(row)

    outdf = pd.DataFrame(rows).sort_values("n_agents")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    outdf.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")
    return outdf


if __name__ == "__main__":
    aggregate()
