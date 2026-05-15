#!/usr/bin/env python3
"""PrivMAS sweep runner.

Runs PrivMAS over a dataset while sweeping the number of specialist agents.
Writes a raw per-run/per-sample CSV, then generates a tradeoff plot:
privacy quality vs coordination overhead.

This directly supports the research question:
- "What is the optimal number of agents that balances performance improvement
   and coordination overhead in privacy-focused multi-agent systems?"

Typical usage
-------------
python experiments/run_privmas_sweep.py \
  --config config.yaml \
  --data-path data/real_long_data.csv \
  --n-values 1,2,3 \
  --max-rows 3

Outputs
-------
- results/privmas_runs.csv
- results/plots/privmas/privacy_quality_vs_overhead.png
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

# Ensure project root is on sys.path when running as a script.
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import load_config

# NOTE: PrivMAS imports are intentionally *lazy* (imported inside `run_sweep`).
# Importing `graph.workflow` triggers spaCy model loading via `src/single_agent.py`,
# which is expensive and noisy for `--help`.


def _parse_int_list(value: str) -> List[int]:
    """Parse comma/space-separated integers."""

    if not value:
        return []

    raw = value.replace(",", " ").split()
    out: List[int] = []
    for part in raw:
        try:
            out.append(int(part))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid int in list: {part!r}") from e
    return out


def _apply_equal_role_counts(cfg: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Return a copy of cfg with detector/analyzer/validator counts all set to n."""

    c = copy.deepcopy(cfg)
    c.setdefault("agents", {})
    c["agents"].setdefault("roles", {})
    for role in ("detector", "analyzer", "validator"):
        c["agents"]["roles"].setdefault(role, {})
        c["agents"]["roles"][role]["count"] = int(n)
    return c


def run_sweep(
    *,
    config_path: str,
    data_path: str,
    n_values: List[int],
    max_rows: Optional[int],
    runs_per_n: int,
    out_csv: str,
    out_plot: str,
    overhead_metric: str,
    quality_metric: str,
) -> None:
    # Lazy imports to avoid loading heavy spaCy/PyTorch stacks for `--help`.
    from evaluation.metrics import compute_all_metrics
    from graph.workflow import run_privmas_once
    from visualization.privmas_plots import plot_privacy_quality_vs_overhead

    cfg_base: Dict[str, Any] = load_config(config_path)

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if "text" not in df.columns:
        raise ValueError(f"CSV must contain a 'text' column: {data_path}")

    if max_rows is not None:
        df = df.head(int(max_rows))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    rows: List[Dict[str, Any]] = []

    total_jobs = max(len(n_values), 1) * max(runs_per_n, 1) * max(len(df), 1)
    done = 0

    for n in n_values:
        if n <= 0:
            continue

        cfg = _apply_equal_role_counts(cfg_base, n)
        total_specialists = 3 * int(n)

        for run_idx in range(int(runs_per_n)):
            for sample_idx, row in df.iterrows():
                done += 1
                text = str(row.get("text", ""))
                label = row.get("label")

                state = run_privmas_once(
                    text=text,
                    label=None if label is None else str(label),
                    run_id=f"sample={sample_idx}|n={n}|run={run_idx}",
                    config=cfg,
                )

                metrics = compute_all_metrics(state)

                coord = metrics.get("coordination", {})
                audit = metrics.get("privacy_audit", {})
                llm = metrics.get("llm_tokens", {})

                residual_email = int(audit.get("residual_email_count") or 0)
                residual_phone = int(audit.get("residual_phone_count") or 0)

                rows.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "config_path": config_path,
                        "data_path": data_path,
                        "sample_id": int(sample_idx),
                        "run_idx": int(run_idx),
                        "n_per_role": int(n),
                        "detectors": int(n),
                        "analyzers": int(n),
                        "validators": int(n),
                        "total_specialists": int(total_specialists),
                        "final_strategy": state.final_strategy,
                        "e2e_ms": (state.timings_ms or {}).get("e2e_ms"),
                        "message_count": coord.get("message_count"),
                        "message_bytes": coord.get("message_bytes"),
                        "message_tokens_approx": coord.get("message_tokens_approx"),
                        "placeholder_count": audit.get("placeholder_count"),
                        "residual_email_count": residual_email,
                        "residual_phone_count": residual_phone,
                        "residual_pii_count": residual_email + residual_phone,
                        "llm_calls": (llm.get("total_approx") or {}).get("calls"),
                        "llm_total_tokens_approx": (llm.get("total_approx") or {}).get("total_tokens"),
                        "llm_total_tokens_exact": (llm.get("total_exact") or {}).get("total_tokens"),
                        "error_count": len(state.errors or []),
                    }
                )

                if done % 1 == 0:
                    print(f"[{done}/{total_jobs}] n={n} run={run_idx} sample={sample_idx} done")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(out_df)} rows)")

    # Plot: privacy quality vs coordination overhead tradeoff
    plot_privacy_quality_vs_overhead(
        out_csv,
        out_plot,
        overhead_metric=overhead_metric,
        quality_metric=quality_metric,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PrivMAS sweep runner (agent-count ablations)")
    parser.add_argument("--config", default="config.yaml", help="Path to PrivMAS config YAML")
    parser.add_argument(
        "--data-path",
        default="data/real_long_data.csv",
        help="Path to CSV with a 'text' column (and optional 'label')",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit")
    parser.add_argument(
        "--n-values",
        type=_parse_int_list,
        default=_parse_int_list("1,2,3"),
        help="Comma/space-separated sweep values; applied equally to detector/analyzer/validator",
    )
    parser.add_argument("--runs-per-n", type=int, default=1, help="Repeat each n for latency variance")
    parser.add_argument("--out-csv", default="results/privmas_runs.csv", help="Where to write raw sweep rows")
    parser.add_argument(
        "--out-plot",
        default="results/plots/privmas/privacy_quality_vs_overhead.png",
        help="Where to write the tradeoff plot",
    )
    parser.add_argument(
        "--overhead-metric",
        default="message_bytes",
        choices=["message_count", "message_bytes", "message_tokens_approx", "e2e_ms"],
        help="X-axis metric for the tradeoff plot",
    )
    parser.add_argument(
        "--quality-metric",
        default="residual_pii_count",
        choices=["residual_pii_count", "residual_email_count", "residual_phone_count"],
        help="Y-axis metric for the tradeoff plot",
    )

    args = parser.parse_args()

    if not args.n_values:
        raise SystemExit("--n-values must contain at least one integer")

    run_sweep(
        config_path=args.config,
        data_path=args.data_path,
        n_values=args.n_values,
        max_rows=args.max_rows,
        runs_per_n=args.runs_per_n,
        out_csv=args.out_csv,
        out_plot=args.out_plot,
        overhead_metric=args.overhead_metric,
        quality_metric=args.quality_metric,
    )


if __name__ == "__main__":
    main()
