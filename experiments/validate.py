#!/usr/bin/env python3
"""Validation script to verify IoU accuracy and metric correctness."""

import sys
import os

# ensure project root is on sys.path when running as script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import load_demo_dataset
from orchestrator.orchestrator import run_pipeline
from utils.metrics import evaluate_run


TOL = 1e-6


def validate_sample(sample, n_agents=4, iou_threshold=0.5):
    text = sample["text"]
    gt = sample.get("entities", [])
    final_text, metrics, agent_results = run_pipeline(text, n_agents=n_agents)

    # Evaluate accuracy via IoU
    eval_res = evaluate_run(agent_results, gt, iou_threshold=iou_threshold)

    # tokens_total check
    tokens_sum = sum(getattr(r, "tokens_processed", 0) for r in agent_results)
    if tokens_sum != metrics.get("tokens_total"):
        raise AssertionError(f"tokens_total mismatch: metrics {metrics.get('tokens_total')} vs computed {tokens_sum}")

    # critical_path check
    exec_times = [getattr(r, "processing_time", 0.0) for r in agent_results]
    critical_path = max(exec_times) if exec_times else 0.0
    if abs(critical_path - metrics.get("critical_path", 0.0)) > TOL:
        raise AssertionError(f"critical_path mismatch: metrics {metrics.get('critical_path')} vs computed {critical_path}")

    # coordination tax check
    expected_coord = metrics.get("total_latency") - critical_path
    if abs(expected_coord - metrics.get("coordination_tax", 0.0)) > TOL:
        raise AssertionError(f"coordination_tax mismatch: metrics {metrics.get('coordination_tax')} vs computed {expected_coord}")

    # sync_delay check
    arrival_times = [getattr(r, "arrival_time", getattr(r, "end_time", 0.0)) for r in agent_results]
    if arrival_times:
        sync_delay = max(arrival_times) - min(arrival_times)
    else:
        sync_delay = 0.0
    if abs(sync_delay - metrics.get("sync_delay", 0.0)) > TOL:
        raise AssertionError(f"sync_delay mismatch: metrics {metrics.get('sync_delay')} vs computed {sync_delay}")

    # Check predicted entity spans have integer start/end
    for r in agent_results:
        for p in getattr(r, "predicted_entities", []):
            if not isinstance(p.get("start"), int) or not isinstance(p.get("end"), int):
                raise AssertionError(f"predicted entity start/end not ints: {p}")

    print("Validation OK: tokens_total, critical_path, coordination_tax, sync_delay consistent.")
    print("Evaluation:", eval_res)
    return True


def main():
    dataset = load_demo_dataset()
    all_ok = True
    for i, sample in enumerate(dataset):
        try:
            print(f"Validating sample {i}...")
            validate_sample(sample, n_agents=4, iou_threshold=0.5)
        except AssertionError as e:
            print(f"Validation FAILED for sample {i}: {e}")
            all_ok = False
    if not all_ok:
        print("Validation failed.")
        sys.exit(2)
    print("All validations passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
