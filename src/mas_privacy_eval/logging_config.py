"""Logging configuration utilities."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str) -> None:
    """Configure stdlib logging once for the whole app."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
