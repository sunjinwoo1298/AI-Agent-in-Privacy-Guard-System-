"""A generalist agent that performs a sequence of masking strategies."""

from typing import Any, Dict, Optional

from privacy.masker import PrivacyMasker
from core.state import MaskingResult, MaskingStrategy

class GeneralistAgent:
    """A single agent that can apply a specific masking strategy."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, assigned_strategy: Optional[MaskingStrategy] = None):
        self.config = config or {}
        self.masker = PrivacyMasker(self.config)
        
        agent_cfg = (self.config.get("agents") or {}).get("generalist", {})
        # The assigned strategy overrides any default sequence.
        self.strategy = assigned_strategy or agent_cfg.get("default_strategy", "spacy_plus")

    def process_chunk(self, text: str, chunk_id: int) -> Dict[str, Any]:
        """
        Processes a single chunk of text by applying its assigned masking strategy.
        """
        agent_id = f"generalist_{chunk_id}"
        
        result: MaskingResult = self.masker.mask(
            text,
            agent_id=agent_id,
            role="generalist",
            strategy=self.strategy,
        )

        return {
            "chunk_id": chunk_id,
            "masked_text": result.masked_text,
            "latency_ms": result.latency_ms,
            "error": result.error,
            "strategy": self.strategy, # Include strategy in result
            "detected_entities": result.detected_entities,
        }
