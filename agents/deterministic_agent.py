"""Deterministic privacy agent: detects and masks PII deterministically."""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# Regex detectors
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"\b\d{10}\b")
# Heuristic: require two capitalized words (reduce false positives like 'Email' or city names)
NAME_HEURISTIC_REGEX = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b")


@dataclass
class AgentResult:
    agent_id: int
    masked_items: List[Dict[str, Any]] = field(default_factory=list)
    predicted_entities: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    processing_time: float = 0.0
    tokens_processed: int = 0
    arrival_time: Optional[float] = None


class DeterministicAgent:
    """Agent that detects PII deterministically and masks it.

    The agent returns predicted entity spans before masking (absolute offsets),
    so accuracy can be computed later.
    """

    def __init__(self, agent_id: int, nlp: Optional[Any] = None):
        self.agent_id = int(agent_id)
        self.nlp = nlp

    def _merge_spans(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not spans:
            return []
        spans = sorted(spans, key=lambda s: (s["start"], -s["end"]))
        merged = [spans[0].copy()]
        for s in spans[1:]:
            last = merged[-1]
            if s["start"] <= last["end"]:
                # overlap: extend end
                last["end"] = max(last["end"], s["end"])
            else:
                merged.append(s.copy())
        return merged

    def process(self, shard: List[Dict[str, Any]]) -> AgentResult:
        start = time.perf_counter()
        result = AgentResult(agent_id=self.agent_id)

        for sent in shard:
            idx = sent["index"]
            text = sent["text"]
            abs_start = sent["abs_start"]
            # collect predicted spans (absolute)
            predicted = []

            # emails
            for m in EMAIL_REGEX.finditer(text):
                predicted.append({"label": "EMAIL", "start": abs_start + m.start(), "end": abs_start + m.end(), "text": m.group()})

            # phones
            for m in PHONE_REGEX.finditer(text):
                predicted.append({"label": "PHONE", "start": abs_start + m.start(), "end": abs_start + m.end(), "text": m.group()})

            # names: prefer spaCy NER when available
            if self.nlp is not None:
                try:
                    doc = self.nlp(text)
                    for ent in doc.ents:
                        if ent.label_ in ("PERSON", "GPE"):
                            predicted.append({"label": "NAME", "start": abs_start + ent.start_char, "end": abs_start + ent.end_char, "text": ent.text})
                except Exception:
                    pass
            else:
                # heuristic: capitalized words
                for m in NAME_HEURISTIC_REGEX.finditer(text):
                    predicted.append({"label": "NAME", "start": abs_start + m.start(), "end": abs_start + m.end(), "text": m.group()})

            # merge overlapping spans deterministically
            merged = self._merge_spans([
                {"start": s["start"], "end": s["end"], "label": s.get("label", "NAME"), "text": s.get("text", "")}
                for s in predicted
            ])

            # build masked sentence by replacing spans from left to right
            rel_masked_parts = []
            last = 0
            for sp in merged:
                rel_s = sp["start"] - abs_start
                rel_e = sp["end"] - abs_start
                if rel_s < last:
                    # overlapping / already consumed
                    last = max(last, rel_e)
                    continue
                rel_masked_parts.append(text[last:rel_s])
                tag = "[NAME]"
                if sp["label"] == "EMAIL":
                    tag = "[EMAIL]"
                elif sp["label"] == "PHONE":
                    tag = "[PHONE]"
                rel_masked_parts.append(tag)
                last = rel_e
            rel_masked_parts.append(text[last:])
            masked_sentence = "".join(rel_masked_parts).strip()

            # record
            result.masked_items.append({"index": idx, "masked_sentence": masked_sentence})
            # store predicted entity spans with absolute offsets using keys `start`/`end`
            for sp in merged:
                result.predicted_entities.append({"label": sp.get("label", "NAME"), "start": sp["start"], "end": sp["end"], "text": sp.get("text", "")})

            result.tokens_processed += len(text.split())

        end = time.perf_counter()
        result.start_time = start
        result.end_time = end
        result.processing_time = end - start
        return result


__all__ = ["DeterministicAgent", "AgentResult"]
