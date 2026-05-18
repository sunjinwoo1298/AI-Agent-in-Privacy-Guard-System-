"""Project entry point.

Runs an empirical multi-agent privacy evaluation sweep and writes results to disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List


# Allow running from a src-layout project without installing the package.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mas_privacy_eval.config import AppConfig, ExperimentConfig, ModelConfig
from mas_privacy_eval.experiment.runner import run_experiment
from mas_privacy_eval.logging_config import configure_logging


logger = logging.getLogger(__name__)


def _parse_int_list(csv: str) -> List[int]:
    items = [s.strip() for s in csv.split(",") if s.strip()]
    return [int(x) for x in items]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Empirical multi-agent privacy evaluation")

    # Model
    p.add_argument("--model", dest="model_name", default=ModelConfig().model_name)
    p.add_argument("--max-new-tokens", type=int, default=ModelConfig().max_new_tokens)
    p.add_argument("--temperature", type=float, default=ModelConfig().temperature)
    p.add_argument("--top-p", type=float, default=ModelConfig().top_p)
    p.add_argument("--no-sampling", action="store_true", help="Disable sampling for generation")

    # Experiment
    p.add_argument("--agent-counts", type=str, default=",".join(map(str, ExperimentConfig().agent_counts)))
    p.add_argument("--topologies", type=str, default=",".join(ExperimentConfig().topologies))
    p.add_argument("--trials", type=int, default=ExperimentConfig().n_trials)
    p.add_argument("--samples", type=int, default=ExperimentConfig().n_samples_per_trial)
    p.add_argument("--dry-run", action="store_true", help="Run a tiny sweep for verification")
    p.add_argument("--seed", type=int, default=ExperimentConfig().master_seed)
    p.add_argument("--output-dir", type=Path, default=ExperimentConfig().output_dir)

    # Output controls
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--skip-stats", action="store_true")

    # Logging
    p.add_argument("--log-level", type=str, default=AppConfig().log_level)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    app_config = AppConfig(
        log_level=args.log_level,
        model=ModelConfig(
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.no_sampling,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
        experiment=ExperimentConfig(
            agent_counts=_parse_int_list(args.agent_counts),
            topologies=[s.strip() for s in args.topologies.split(",") if s.strip()],
            n_trials=args.trials,
            n_samples_per_trial=args.samples,
            dry_run=args.dry_run,
            master_seed=args.seed,
            output_dir=args.output_dir,
        ),
    )

    logger.info("Starting experiment")
    logger.info("Model: %s", app_config.model.model_name)
    logger.info("Topologies: %s", app_config.experiment.topologies)
    logger.info("Agent counts: %s", app_config.experiment.agent_counts)
    run_experiment(app_config, make_plots=not args.skip_plots, run_stats=not args.skip_stats)
    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
