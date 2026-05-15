"""Deterministic sharding utilities (sentence splitting + offsets)."""

import re
from typing import List, Dict, Any


_SENT_PATTERN = re.compile(r".+?(?:[.!?](?:\s|$)|\Z)", re.DOTALL)


def split_text_into_sentences(text: str) -> List[Dict[str, Any]]:
    """Split `text` into sentences and return list of dicts with absolute offsets.

    Returns list of {index, text, abs_start, abs_end}
    """
    sentences = []
    idx = 0
    for m in _SENT_PATTERN.finditer(text):
        s = m.group().strip()
        if not s:
            continue
        start = m.start()
        end = m.end()
        sentences.append({"index": idx, "text": s, "abs_start": start, "abs_end": end})
        idx += 1
    return sentences


__all__ = ["split_text_into_sentences"]
