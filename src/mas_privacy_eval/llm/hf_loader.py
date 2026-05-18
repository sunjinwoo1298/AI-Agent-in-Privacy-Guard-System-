"""Hugging Face model loader."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedHFModel:
    model: object
    tokenizer: object


def load_hf_chat_model(model_name: str) -> LoadedHFModel:
    """Load a chat-capable causal LM + tokenizer.

Quantization is intentionally not enabled by default to keep this project
cross-platform (e.g., bitsandbytes is often problematic on Windows).
"""

    logger.info("Loading Hugging Face model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    logger.info("Model loaded")
    return LoadedHFModel(model=model, tokenizer=tokenizer)
