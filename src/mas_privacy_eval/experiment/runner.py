"""Experiment sweep runner.

Runs multi-agent topology sweeps and writes metrics + artifacts to disk.
"""

from __future__ import annotations

import dataclasses
import logging
import zlib
from pathlib import Path
from typing import Dict, List

import pandas as pd

from mas_privacy_eval.agents.provider import HFRealAgentProvider, HeuristicAgentProvider
from mas_privacy_eval.analysis.summary import build_bootstrap_ci_table, build_summary_table, write_stats_report
from mas_privacy_eval.config import AppConfig
from mas_privacy_eval.data.loader import PrivacyDatasetLoader
from mas_privacy_eval.data.sampling import stratified_sample_by_difficulty
from mas_privacy_eval.io.files import append_csv_row, append_jsonl, ensure_dir, reset_file
from mas_privacy_eval.llm.hf_loader import load_hf_chat_model
from mas_privacy_eval.metrics.core import ExperimentMetrics, compute_metrics
from mas_privacy_eval.pipelines.blackboard import BlackboardPipeline
from mas_privacy_eval.pipelines.hierarchical import HierarchicalPipeline
from mas_privacy_eval.pipelines.parallel import ParallelPipeline
from mas_privacy_eval.pipelines.sequential import SequentialPipeline
from mas_privacy_eval.viz.plotting import save_results_plot


logger = logging.getLogger(__name__)


TOPOLOGY_FACTORIES = {
    "sequential": SequentialPipeline,
    "parallel": ParallelPipeline,
    "hierarchical": HierarchicalPipeline,
    "blackboard": BlackboardPipeline,
}


def _stable_topology_hash(name: str) -> int:
    return int(zlib.adler32(name.encode("utf-8")) % 1000)


def run_experiment(app_config: AppConfig, *, make_plots: bool, run_stats: bool) -> None:
    cfg = app_config
    out_dir: Path = cfg.experiment.output_dir
    ensure_dir(out_dir)

    metrics_path = out_dir / "metrics.csv"
    raw_path = out_dir / "raw_results.jsonl"
    summary_path = out_dir / "summary.csv"
    plot_path = out_dir / "mas_privacy_results.png"
    stats_path = out_dir / "stats.txt"

    reset_file(metrics_path)
    reset_file(raw_path)

    if cfg.experiment.dry_run:
        logger.info("Dry-run enabled: using heuristic agents (no HF model load)")
        agent_provider = HeuristicAgentProvider(seed=cfg.experiment.master_seed)
        title_model = "Heuristic dry-run"
    else:
        loaded = load_hf_chat_model(cfg.model.model_name)
        agent_provider = HFRealAgentProvider(
            hf_model=loaded.model,
            hf_tokenizer=loaded.tokenizer,
            model_cfg=cfg.model,
            verbose=False,
        )
        title_model = cfg.model.model_name

    all_metrics: List[ExperimentMetrics] = []
    all_raw_results: List[Dict] = []

    n_samples = 2 if cfg.experiment.dry_run else int(cfg.experiment.n_samples_per_trial)

    logger.info(
        "Sweep: topologies=%s agent_counts=%s trials=%d samples/trial=%d",
        cfg.experiment.topologies,
        cfg.experiment.agent_counts,
        cfg.experiment.n_trials,
        n_samples,
    )

    for topology in cfg.experiment.topologies:
        topology = topology.strip().lower()
        factory = TOPOLOGY_FACTORIES.get(topology)
        if factory is None:
            logger.warning("Unknown topology '%s'; skipping", topology)
            continue

        for n in cfg.experiment.agent_counts:
            for trial in range(int(cfg.experiment.n_trials)):
                seed = (
                    int(cfg.experiment.master_seed)
                    + _stable_topology_hash(topology)
                    + int(n) * 100
                    + int(trial) * 7
                )

                # Sample dataset for this trial
                n_curated = max(int(cfg.dataset.n_curated), int(n_samples) * 3)
                trial_loader = PrivacyDatasetLoader(seed=seed)
                dataset = trial_loader.load(
                    n_curated=n_curated,
                    n_hf=int(cfg.dataset.n_hf),
                    max_hf_text_chars=int(cfg.dataset.max_hf_text_chars),
                    enable_augmentation=bool(cfg.dataset.enable_augmentation),
                )
                trial_data = stratified_sample_by_difficulty(dataset, n=n_samples, seed=seed)

                pipeline = factory(n_agents=int(n), agent_provider=agent_provider)

                results = []
                for sample in trial_data:
                    try:
                        pr = pipeline.run_sample(sample)
                        results.append(pr)

                        raw_rec = {
                            "topology": topology,
                            "n_agents": int(n),
                            "trial": int(trial),
                            "seed": int(seed),
                            "sample_id": int(sample.sample_id),
                            "true_label": int(sample.true_label),
                            "pred": pr.final_prediction,
                            "confidence": float(pr.final_confidence),
                            "latency_ms": float(pr.total_latency_ms),
                            "tokens": int(pr.total_tokens),
                            "disagreement": bool(pr.disagreement),
                            "escalated": bool(pr.escalated),
                            "parse_failures": int(pr.parse_failures),
                            "parse_retries": int(pr.parse_retries),
                            "difficulty": str(sample.difficulty),
                            "category": str(sample.category),
                        }
                        all_raw_results.append(raw_rec)
                        append_jsonl(raw_path, raw_rec)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Error on sample %s (topology=%s n=%s trial=%s): %s",
                            sample.sample_id,
                            topology,
                            n,
                            trial,
                            exc,
                        )

                if not results:
                    logger.warning("No results for topology=%s n=%s trial=%s", topology, n, trial)
                    continue

                m = compute_metrics(results, topology, int(n), int(trial), int(seed))
                all_metrics.append(m)
                append_csv_row(metrics_path, dataclasses.asdict(m))

                logger.info(
                    "%-12s N=%d trial=%d f1=%.3f prec=%.3f rec=%.3f lat=%.1fms tok=%.1f disagr=%.1f%% fail=%.1f%%",
                    topology,
                    int(n),
                    int(trial),
                    m.f1,
                    m.precision,
                    m.recall,
                    m.mean_latency_ms,
                    m.mean_tokens,
                    100.0 * m.disagreement_rate,
                    100.0 * m.parse_failure_rate,
                )

    if not all_metrics:
        logger.error("No metrics computed; nothing to write")
        return

    df_metrics = pd.DataFrame([dataclasses.asdict(m) for m in all_metrics])
    df_raw = pd.DataFrame(all_raw_results)
    df_summary = build_summary_table(df_metrics)
    df_ci = build_bootstrap_ci_table(df_metrics)

    df_summary.to_csv(summary_path, index=False)

    if run_stats:
        write_stats_report(df_metrics, df_raw, stats_path)

    if make_plots:
        title = (
            "Empirical Multi-Agent Privacy Evaluation\n"
            f"Backend: {title_model} · N={min(cfg.experiment.agent_counts)}–{max(cfg.experiment.agent_counts)} · "
            f"Topologies={', '.join(sorted(set(df_metrics['topology'].tolist())))} · Trials={cfg.experiment.n_trials}"
        )
        save_results_plot(
            df_summary=df_summary,
            df_ci=df_ci,
            agent_counts=[int(x) for x in cfg.experiment.agent_counts],
            output_path=plot_path,
            title=title,
        )

    logger.info("Wrote outputs to: %s", out_dir)
