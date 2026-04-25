"""Aggregator: wait for agents and merge their masked shards deterministically."""

from typing import List, Dict, Any


def merge_masked_sentences(agent_results: List[Any]) -> str:
    """Merge masked sentence fragments from agents and return final text.

    agent_results: list of AgentResult objects, each containing `masked_items` where
    each entry is a dict with `index` and `masked_sentence`.
    """
    items = []
    for r in agent_results:
        for it in r.masked_items:
            items.append((it["index"], it["masked_sentence"]))

    # sort by original sentence index to preserve original order
    items_sorted = sorted(items, key=lambda x: x[0])
    sentences = [s for idx, s in items_sorted]
    final_text = " ".join(sentences)
    return final_text


__all__ = ["merge_masked_sentences"]
