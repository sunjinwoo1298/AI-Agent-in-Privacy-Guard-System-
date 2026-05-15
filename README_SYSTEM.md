# Deterministic Multi-Agent PII Detection & Masking — System Overview

This repository implements a deterministic multi-agent (MAS) pipeline that detects and masks PII (names, emails, phone numbers) in text, with an experiment framework to measure parallel performance (ThreadPool vs ProcessPool), instrumentation, and accuracy metrics.

This README explains the system end-to-end, the experimental setup, reproducibility controls, and a short example showing how the pipeline runs.

---

## 1) High-level overview

- Input: a text sample (single document or sentence sequence).
- Sharding: sentences are split deterministically and assigned to agents using round-robin sharding (`sentence_index % n_agents`).
- Agents: each agent runs deterministic PII detectors (regex for EMAIL/PHONE; NAME heuristic or optional spaCy NER). Agents return `predicted_entities` (spans) and a masked fragment.
- Orchestrator: dispatches shards to workers (ThreadPool or ProcessPool), collects per-agent timings and outputs, computes system-level metrics, and merges masked fragments back by sentence index.
- Evaluation: predicted entities are IoU-matched to ground-truth spans → TP/FP/FN → precision/recall/F1.
- Storage & analysis: raw per-run rows are appended to CSV; aggregation computes mean/std/p95; plotting + PDF report generation produce visual summaries.

Core scripts and modules (important paths):

- `main.py` — demo runner (prints original/masked text, metrics, appends run to `results/logs.csv`).
- `data/dataset.py` — synthetic dataset generator (`seed`-able presets: small/medium/large). Use this to reproduce inputs.
- `orchestrator/orchestrator.py` — shard/dispatch/collect/merge pipeline (ThreadPool/ProcessPool selectable).
- `agents/deterministic_agent.py` — deterministic agent logic (regex + optional spaCy). Masks using tokens `[EMAIL]`, `[PHONE]`, `[NAME]`.
- `aggregator/aggregator.py` — merge masked sentence fragments deterministically.
- `utils/` — includes `sharding.py`, `timing.py`, `metrics.py` (IoU matcher + evaluator), `storage.py` (CSV append).
- `experiments/` — YAML-driven runner (`run_experiments.py`) and presets (`config_heavy_thread.yaml`, `config_heavy_process.yaml`).
- `experiments/aggregate_results.py` — computes mean/std/p95 per `n_agents`.
- `visualization/plots.py` — summary and comparison plotting utilities.
- `visualization/generate_report.py` — creates a combined PDF with `results/analysis.txt` and comparison images.

---

## 2) Determinism & reproducibility

- Dataset generation is seeded (default `seed: 42` in YAML configs) so the same synthetic examples are produced between runs.
- Sentence splitting and sharding are deterministic (sentence order and the `sentence_index % n_agents` mapping are fixed).
- Agent logic is deterministic: regex detectors and a `two-word capitalized` heuristic for `NAME` ensure stable behavior across runs. When using spaCy, model loading can be toggled with `use_spacy` but model behavior remains deterministic for the same inputs.

To fully reproduce an experiment, set the `seed` field in the YAML config under `experiments/` and use the provided config file (e.g. `experiments/config_heavy_process.yaml`).

---

## 3) Metrics collected

- `total_latency` — end-to-end runtime for the sample (seconds).
- `per_agent_times` — execution time per agent (used to compute critical path).
- `critical_path` (`T_inf`) — the longest agent compute time for the sample.
- `coordination_tax` — `total_latency - critical_path` (time spent in dispatch/merge/waiting/overhead).
- `efficiency` — `critical_path / total_latency` (closer to 1 ⇒ less coordination overhead).
- `sync_delay` — merge/wait-specific measurement (keeps analysis explicit).
- `tokens_total` and `total_throughput` (tokens/sec) — throughput metrics.
- Accuracy: `precision`, `recall`, `f1` from IoU-based span matching (see `utils/metrics.py`).

All raw per-run rows are appended to the CSV configured in the experiment YAML (e.g. `results/logs_heavy_process.csv`), and `experiments/aggregate_results.py` computes per-`n_agents` summaries.

---

## 4) Experimental setup (what the YAML controls)

Typical fields in `experiments/config_*.yaml`:

- `n_values`: list of `n` agent counts to sweep (e.g. `[1,2,4,8,16]`).
- `runs_per_n`: number of repetitions per `n`.
- `dataset`: preset (small/medium/large), `sample_count`, `tokens_per_sample`, `entity_density`.
- `iou_threshold`: IoU threshold for matching spans.
- `use_spacy`: bool — whether agents will use spaCy for `NAME` detection.
- `use_process_pool`: bool — backend choice.
- `seed`: RNG seed to make dataset and sharding deterministic.
- `results_csv`: path to append raw per-run rows.

Example (heavy process config): `experiments/config_heavy_process.yaml`.

---

## 5) How to run — quick commands

Run a quick demo (prints outputs and logs a run):

```bash
python main.py --n 4 --sample 0
```

Run a heavy-process experiment (async/long-running):

```bash
python -u experiments/run_experiments.py experiments/config_heavy_process.yaml
```

Run the equivalent heavy-thread experiment:

```bash
python -u experiments/run_experiments.py experiments/config_heavy_thread.yaml
```

Aggregate raw CSV to summary (example used in this repo):

```bash
python -c "import experiments.aggregate_results as ar; ar.aggregate('results/logs_heavy_process.csv','results/summary_heavy_process.csv')"
python -c "import experiments.aggregate_results as ar; ar.aggregate('results/logs_heavy_thread.csv','results/summary_heavy_thread.csv')"
```

Generate plots and comparison PDF (examples used by the repo):

```bash
python -c "import visualization.plots as vp; vp.plot_all('results/summary_heavy_process.csv','results/plots/heavy_process')"
python -c "import visualization.plots as vp; vp.compare_all('results/summary_heavy_thread.csv','results/summary_heavy_process.csv','results/plots/comparison')"
python visualization/generate_report.py
```

Notes:
- For `use_spacy: true` you must install the model: `pip install spacy && python -m spacy download en_core_web_sm`.
- ProcessPool runs incur per-process startup and data marshalling costs; for fair comparisons warm-start processes or increase per-agent workload (tokens/sample) so compute dominates coordination.

---

## 6) How the pipeline works — a short example

1. The orchestrator reads the input sample and splits into sentences deterministically.
2. Sentences are assigned to `n` agents by index modulo `n`.
3. The orchestrator dispatches jobs to a `ThreadPoolExecutor` (or `ProcessPoolExecutor` if `use_process_pool=True`).
4. Each agent runs `DeterministicAgent.process()` returning masked text and predicted spans.
5. Orchestrator collects timings, calculates `T_inf` and `coordination_tax`, merges masked fragments using `aggregator.merge_masked_sentences()`.
6. If ground truth entities are available, the orchestrator evaluates predictions with IoU matching and appends the run to `results/*.csv`.

Example demo (what you will see):

```
Running demo pipeline on sample 0 with 4 agents

--- Original Text ---
"John Doe (john.doe@example.com) called +1-555-123-4567 about the contract."

--- Masked Output ---
"[NAME] ([EMAIL]) called [PHONE] about the contract."

--- Metrics ---
total_latency: 0.0021
critical_path: 0.0015
coordination_tax: 0.0006
efficiency: 0.714

--- Accuracy ---
precision: 0.90
recall: 0.88
f1: 0.89
```

The example above is illustrative — precise values depend on dataset preset and `n`.

---

## 7) Useful files & artifacts produced by experiments

- `results/logs.csv`, `results/logs_heavy_thread.csv`, `results/logs_heavy_process.csv` — raw per-run rows.
- `results/summary_heavy_thread.csv`, `results/summary_heavy_process.csv` — aggregated mean/std/p95 per `n_agents`.
- `results/plots/` — summary and comparison PNGs.
- `results/plots/comparison/report.pdf` — combined PDF report (analysis + plots).
- `results/analysis.txt` — textual analysis generated by `analysis/analyze_results.py`.

---

## 8) Recommendations & next steps

- For truly CPU-bound workloads, increase `tokens_per_sample` in the YAML so agent compute dominates coordination — ProcessPool then shows benefit.
- Profile the `coordination_tax` contribution and consider batching many sentences per agent to reduce communication overhead.
- Add integration tests to validate deterministic behavior across environments.

If you want, I can add a small `Makefile`/script to run the full reproduce pipeline (generate dataset → run experiments → aggregate → plot → PDF). Ask and I will add it.
