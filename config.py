"""Project configuration loader.

This module fixes the existing import in `src/single_agent.py`:

    from config import GROQ_API_KEY

Design goals:
- Safe to commit: no secrets are stored here.
- Loads configuration from (1) environment variables, (2) optional `.env`,
  and (3) optional `config.yaml`.
- Provides a simple, stable API for the rest of the project.

Notes
-----
- `.env` is optional and is intended for secrets like GROQ_API_KEY.
- `config.yaml` is optional and is intended for experiment settings (agent counts, topology, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import os

import yaml
from dotenv import load_dotenv


# Load `.env` at import time so `GROQ_API_KEY` is available immediately for
# `src/single_agent.py` (which imports `GROQ_API_KEY` at module import time).
# `override=False` ensures real environment variables take precedence.
load_dotenv(override=False)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _safe_read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        # Config should never crash the app; fall back to defaults.
        return {}


def load_config(config_path: Optional[Union[str, os.PathLike[str]]] = None) -> Dict[str, Any]:
    """Load the project configuration.

    Precedence:
    1) Environment variables (e.g., GROQ_API_KEY)
    2) `.env` (if present)
    3) YAML file (default: config.yaml)

    Returns
    -------
    dict
        Nested configuration dictionary.
    """

    # Load .env if present (no-op if missing)
    load_dotenv(override=False)

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    cfg = _safe_read_yaml(path)

    # Attach resolved secrets / env-derived values
    cfg.setdefault("llm", {})
    cfg["llm"].setdefault("api_key_env", "GROQ_API_KEY")

    return cfg


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable with an optional default."""

    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


def get_nested(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a nested value using dot notation (e.g. "agents.roles.detector.count")."""

    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# --- Compatibility constant for existing code ---
# `src/single_agent.py` checks for the exact placeholder string "YOUR_API_KEY".
# Keeping that default prevents accidental network calls with an invalid key.
GROQ_API_KEY: str = get_env("GROQ_API_KEY", default="YOUR_API_KEY")
