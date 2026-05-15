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


def shard_text(text: str, num_shards: int) -> List[str]:
    """
    Splits a string into a specified number of roughly equal-sized shards.
    """
    if num_shards <= 0:
        return []
    if num_shards == 1:
        return [text]
    
    text_len = len(text)
    shard_len = text_len // num_shards
    shards = []
    
    start = 0
    for i in range(num_shards):
        end = start + shard_len
        # For the last shard, extend it to the end of the text
        if i == num_shards - 1:
            end = text_len
        shards.append(text[start:end])
        start = end
        
    return shards


__all__ = ["split_text_into_sentences", "shard_text"]
