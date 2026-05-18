"""Central configuration models.

All runtime-tunable parameters live here (CLI overrides these defaults).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ModelConfig:
    """Settings for the Hugging Face generation backend."""

    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    max_new_tokens: int = 400
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass(frozen=True)
class DatasetConfig:
    """Settings for dataset construction."""

    n_curated: int = 60
    n_hf: int = 0
    max_hf_text_chars: int = 500
    enable_augmentation: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment sweep configuration."""

    agent_counts: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    topologies: List[str] = field(default_factory=lambda: ["sequential", "parallel"])
    n_trials: int = 3
    n_samples_per_trial: int = 20
    dry_run: bool = False
    master_seed: int = 42
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class AppConfig:
    """Application configuration bundle."""

    log_level: str = "INFO"
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
