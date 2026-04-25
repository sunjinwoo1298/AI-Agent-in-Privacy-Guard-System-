"""Simple storage helper to append raw per-run metrics to CSV using pandas.

Writes one row per run to `results/logs.csv` (creates file with header if missing).
"""
import os
from datetime import datetime
from typing import Dict, Any

import pandas as pd


def append_run_result(metrics: Dict[str, Any], eval_res: Dict[str, Any], csv_path: str = "results/logs.csv") -> None:
    # compute efficiency if possible (T_inf / L_sys)
    critical_path = metrics.get("critical_path")
    total_latency = metrics.get("total_latency")
    efficiency = None
    if critical_path is not None and total_latency:
        try:
            efficiency = float(critical_path) / float(total_latency)
        except Exception:
            efficiency = None

    row = {
        "run_id": metrics.get("run_id"),
        "n_agents": metrics.get("n_agents"),
        "total_latency": total_latency,
        "critical_path": critical_path,
        "efficiency": efficiency,
        "coordination_tax": metrics.get("coordination_tax"),
        "sync_delay": metrics.get("sync_delay"),
        "total_throughput": metrics.get("total_throughput"),
        "precision": eval_res.get("precision"),
        "recall": eval_res.get("recall"),
        "f1": eval_res.get("f1"),
        "pred_count": eval_res.get("pred_count"),
        "gt_count": eval_res.get("gt_count"),
        "tokens_total": metrics.get("tokens_total"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


__all__ = ["append_run_result"]
