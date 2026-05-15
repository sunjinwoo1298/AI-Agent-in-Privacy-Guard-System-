"""Experiment runner: read YAML config, run experiments, and log raw per-run metrics."""

import os
import sys
import yaml
import time

# ensure project root is on sys.path when running this script directly
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import generate_synthetic_dataset
from orchestrator.orchestrator import run_pipeline
from utils.metrics import evaluate_run
from utils.storage import append_run_result


def load_config(path: str = "experiments/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset(cfg):
    preset = cfg.get("dataset", {}).get("preset", "small")
    sample_count = cfg.get("dataset", {}).get("sample_count", 10)
    tokens_per_sample = cfg.get("dataset", {}).get("tokens_per_sample", 50)
    entity_density = cfg.get("dataset", {}).get("entity_density", 0.05)

    if preset == "small":
        sample_count = sample_count or 10
    elif preset == "medium":
        sample_count = sample_count or 100
    elif preset == "large":
        sample_count = sample_count or 1000

    return generate_synthetic_dataset(count=sample_count, tokens_per_sample=tokens_per_sample, entity_density=entity_density, seed=cfg.get("seed", 42))


def run(cfg_path: str = "experiments/config.yaml"):
    cfg = load_config(cfg_path)
    dataset = resolve_dataset(cfg)
    n_values = cfg.get("n_values", [1, 2, 4, 6, 8, 12, 16])
    runs_per_n = cfg.get("runs_per_n", 5)
    iou_threshold = cfg.get("iou_threshold", 0.5)
    use_spacy = cfg.get("use_spacy", False)

    # Optional: preload spaCy once if requested
    nlp = None
    if use_spacy:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print("spaCy preload failed:", e)
            nlp = None

    total = len(n_values) * runs_per_n * len(dataset)
    cur = 0
    for n in n_values:
        for run_idx in range(runs_per_n):
            for sample in dataset:
                cur += 1
                text = sample["text"]
                final_text, metrics, agent_results = run_pipeline(
                    text, n_agents=n, nlp=nlp, use_spacy=use_spacy, use_process_pool=cfg.get("use_process_pool", False)
                )
                eval_res = evaluate_run(agent_results, sample.get("entities", []), iou_threshold=iou_threshold)
                append_run_result(metrics, eval_res, csv_path=cfg.get("results_csv", "results/logs.csv"))
                print(f"[{cur}/{total}] n={n} run={run_idx} logged")


if __name__ == "__main__":
    # allow optional config path as first arg
    import sys

    cfgp = sys.argv[1] if len(sys.argv) > 1 else "experiments/config.yaml"
    run(cfgp)
