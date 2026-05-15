# Experiment Results — Thread vs Process (Heavy Workload)

This document summarizes the results produced by the heavy workload experiments in this repo (ThreadPool vs ProcessPool). It highlights key findings, points to plots and the combined report, and provides recommended actions based on the current runs.

---

## Key findings (short)

- Coordination overhead dominates this workload: average coordination tax fraction across `n` was ~94.2% (time spent in dispatch/merge/waiting, not compute).
- Total latency increased by ~145.2% from `n=1` to `n=16` for the heavy preset used here.
- Throughput (tokens/sec) decreased as we increased `n` for the chosen workload (tokens_per_sample=750): see detailed table below.
- Accuracy (F1) is stable across `n` (≈ 0.923), indicating masking/evaluation remained deterministic and consistent.
- Best F1/latency trade-off for this experiment was observed at `n=2` (see `results/analysis.txt` for details).

These results indicate that for the chosen synthetic workload (moderate tokens/sample), coordination and IPC overheads from parallelism outweigh per-agent compute gains. For larger per-agent workloads, ProcessPool may show improved scaling.

---

## Numerical summary (from `results/analysis.txt`)

Summary per `n_agents` (mean metrics):

- n=1: total_latency_mean=0.0015s, coordination_tax_mean≈0.0013s (86.1% of total), f1_mean=0.9231
- n=2: total_latency_mean=0.0013s, coordination_tax_mean≈0.0012s (90.6% of total), f1_mean=0.9231
- n=4: total_latency_mean=0.0022s, coordination_tax_mean≈0.0021s (94.8% of total), f1_mean=0.9260
- n=6: total_latency_mean=0.0020s, coordination_tax_mean≈0.0020s (96.3% of total), f1_mean=0.9231
- n=8: total_latency_mean=0.0021s, coordination_tax_mean≈0.0020s (96.3% of total), f1_mean=0.9231
- n=12: total_latency_mean=0.0029s, coordination_tax_mean≈0.0028s (97.3% of total), f1_mean=0.9231
- n=16: total_latency_mean=0.0036s, coordination_tax_mean≈0.0035s (97.7% of total), f1_mean=0.9231

Throughput (tok/s) mean by `n_agents` (from raw logs):

- n=1: 20107.1 tok/s
- n=2: 20742.2 tok/s
- n=4: 13895.1 tok/s
- n=6: 13166.6 tok/s
- n=8: 12542.8 tok/s
- n=12: 8991.4 tok/s
- n=16: 7059.4 tok/s

See `results/analysis.txt` for the full textual report.

---

## Artifacts produced (where to look)

- Combined analysis and plots PDF: `results/plots/comparison/report.pdf`
- Comparison PNGs: `results/plots/comparison/` (files `total_latency_thread_vs_process.png`, `coordination_tax_thread_vs_process.png`, `total_throughput_thread_vs_process.png`, `efficiency_thread_vs_process.png`).
- Aggregated CSVs: `results/summary_heavy_thread.csv`, `results/summary_heavy_process.csv`.
- Raw per-run logs: `results/logs_heavy_thread.csv`, `results/logs_heavy_process.csv`.
- Text analysis: `results/analysis.txt`.

Open the PDF for a concise single-file view containing the textual analysis and visualizations.

---

## Interpretation & recommendations

- For this synthetic workload, the system spends most of the time coordinating rather than computing. That means increasing `n` creates more overhead (dispatch + IPC + merge) than it removes, so the effective throughput and latency worsen for larger `n`.
- If your real workload is CPU-heavy per sample (longer inputs, heavy NER models, or large token counts), increase `tokens_per_sample` in YAML and re-run — ProcessPool is likely to show better scaling there.
- To reduce coordination tax in practice:
  - Batch more sentences per agent (increase shard size).
  - Keep agents warm (preload heavy models inside worker processes before timed measurement).
  - Use shared-memory approaches (if appropriate) to avoid costly serialization for large payloads.

---

## Next steps you can run quickly

Regenerate the report:

```bash
python -u experiments/run_experiments.py experiments/config_heavy_thread.yaml
python -u experiments/run_experiments.py experiments/config_heavy_process.yaml
python -c "import experiments.aggregate_results as ar; ar.aggregate('results/logs_heavy_process.csv','results/summary_heavy_process.csv')"
python -c "import experiments.aggregate_results as ar; ar.aggregate('results/logs_heavy_thread.csv','results/summary_heavy_thread.csv')"
python -c "import visualization.plots as vp; vp.compare_all('results/summary_heavy_thread.csv','results/summary_heavy_process.csv','results/plots/comparison')"
python visualization/generate_report.py
```

If you'd like, I can also add a short script that runs all steps (experiment → aggregate → plot → PDF) reproducibly, or extend the analysis to compute speedup curves and saturation points across multiple `tokens_per_sample` levels.
