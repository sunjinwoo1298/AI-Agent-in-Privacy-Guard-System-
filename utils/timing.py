"""Simple timing helpers."""

import time


def now() -> float:
    return time.perf_counter()


__all__ = ["now"]
