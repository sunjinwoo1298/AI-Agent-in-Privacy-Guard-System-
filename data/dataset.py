"""Dataset utilities and a small deterministic demo dataset.

This module exposes `load_demo_dataset()` which returns a list of samples
with ground truth entities and expected masked outputs.

It also provides a deterministic `generate_synthetic_dataset` stub for later
extension.
"""

import re
from typing import List, Dict, Any

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"\b\d{10}\b")


_DEMO_TEXTS = [
    "John Doe lives in Mumbai. Email: john.doe@gmail.com Phone: 9876543210.",
    "Contact Jane Smith at jane.smith@example.com or 9123456789. She lives in Delhi.",
    "Rohit Sharma moved to Bengaluru. Email: rohit.sharma@example.com Phone: 9988776655.",
]


def _build_sample(text: str) -> Dict[str, Any]:
    entities = []
    for m in EMAIL_REGEX.finditer(text):
        entities.append({"label": "EMAIL", "start": m.start(), "end": m.end(), "text": m.group()})
    for m in PHONE_REGEX.finditer(text):
        entities.append({"label": "PHONE", "start": m.start(), "end": m.end(), "text": m.group()})

    # simple name detection for demo: look for capitalized multi-word names from a small list
    name_candidates = ["John Doe", "Jane Smith", "Rohit Sharma"]
    for nm in name_candidates:
        i = text.find(nm)
        if i >= 0:
            entities.append({"label": "NAME", "start": i, "end": i + len(nm), "text": nm})

    # build expected masked text deterministically by replacing spans from left to right
    spans = sorted(entities, key=lambda s: s["start"])  # start asc
    masked_parts = []
    last = 0
    for sp in spans:
        masked_parts.append(text[last:sp["start"]])
        tag = "[NAME]" if sp["label"] == "NAME" else ("[EMAIL]" if sp["label"] == "EMAIL" else "[PHONE]")
        masked_parts.append(tag)
        last = sp["end"]
    masked_parts.append(text[last:])
    expected = "".join(masked_parts)

    return {"text": text, "entities": entities, "expected_masked": expected}


def load_demo_dataset() -> List[Dict[str, Any]]:
    """Return a deterministic small dataset for demos and testing."""
    return [_build_sample(t) for t in _DEMO_TEXTS]


def generate_synthetic_dataset(count: int = 10, tokens_per_sample: int = 50, entity_density: float = 0.05, complexity: float = 0.5, seed: int = 42) -> List[Dict[str, Any]]:
    """Deterministic synthetic dataset generator (simple stub).

    This function is intentionally simple in the skeleton. It deterministically
    composes sentences from small templates and inserts deterministic entities.
    """
    import random

    r = random.Random(seed)
    names = ["John Doe", "Jane Smith", "Rohit Sharma", "Alice Kumar"]
    cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai"]
    domains = ["example.com", "gmail.com"]

    samples = []
    for i in range(count):
        num_sentences = max(1, tokens_per_sample // 10)
        parts = []
        for s in range(num_sentences):
            name = names[(i + s) % len(names)]
            city = cities[(i + s) % len(cities)]
            email = f"{name.split()[0].lower()}.{i}@{domains[(i + s) % len(domains)]}"
            phone = ''.join(str((i * 7 + s * 3 + k) % 10) for k in range(10))
            if s % 2 == 0:
                parts.append(f"{name} lives in {city}.")
            else:
                parts.append(f"Email: {email} Phone: {phone}.")

        text = " ".join(parts)

        # detect entities using the same simple regex rules
        entities = []
        for m in EMAIL_REGEX.finditer(text):
            entities.append({"label": "EMAIL", "start": m.start(), "end": m.end(), "text": m.group()})
        for m in PHONE_REGEX.finditer(text):
            entities.append({"label": "PHONE", "start": m.start(), "end": m.end(), "text": m.group()})
        for nm in names:
            j = text.find(nm)
            if j >= 0:
                entities.append({"label": "NAME", "start": j, "end": j + len(nm), "text": nm})

        # expected masked (deterministic)
        spans = sorted(entities, key=lambda s: s["start"])  # start asc
        masked_parts = []
        last = 0
        for sp in spans:
            masked_parts.append(text[last:sp["start"]])
            tag = "[NAME]" if sp["label"] == "NAME" else ("[EMAIL]" if sp["label"] == "EMAIL" else "[PHONE]")
            masked_parts.append(tag)
            last = sp["end"]
        masked_parts.append(text[last:])
        expected = "".join(masked_parts)

        samples.append({"text": text, "entities": entities, "expected_masked": expected})

    return samples


__all__ = ["load_demo_dataset", "generate_synthetic_dataset"]
