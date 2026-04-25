#!/usr/bin/env python3
"""Demo runner for the Deterministic Multi-Agent Privacy Pipeline."""

import argparse

from data.dataset import load_demo_dataset
from orchestrator.orchestrator import run_pipeline
from utils.metrics import evaluate_run
from utils.storage import append_run_result
import os


def main():
    parser = argparse.ArgumentParser(description="Deterministic MAS privacy pipeline demo")
    parser.add_argument("--n", type=int, default=2, help="number of agents")
    parser.add_argument("--sample", type=int, default=0, help="dataset sample index")
    args = parser.parse_args()

    dataset = load_demo_dataset()
    sample = dataset[args.sample % len(dataset)]

    print("Running demo pipeline on sample {} with {} agents".format(args.sample, args.n))
    text = sample["text"]

    final_text, metrics, agent_results = run_pipeline(text, n_agents=args.n)

    # Evaluate accuracy via IoU-based span matching
    gt = sample.get("entities", [])
    eval_res = evaluate_run(agent_results, gt, iou_threshold=0.5)

    print("\n--- Original Text ---\n")
    print(text)
    print("\n--- Masked Output ---\n")
    print(final_text)
    print("\n--- Metrics ---\n")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\n--- Accuracy ---\n")
    for k, v in eval_res.items():
        print(f"{k}: {v}")

    # Append raw per-run metrics to CSV
    try:
        append_run_result(metrics, eval_res, csv_path=os.path.join("results", "logs.csv"))
        print("\nRun logged to results/logs.csv")
    except Exception as e:
        print("\nFailed to write run log:", e)


if __name__ == "__main__":
    main()
