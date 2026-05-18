"""Pipeline types shared across topologies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from mas_privacy_eval.agents.types import AgentOutput


class TopologyType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    BLACKBOARD = "blackboard"


@dataclass
class PipelineResult:
    """Full result from running a MAS pipeline on one sample."""

    sample_id: int
    true_label: int
    final_prediction: Optional[int]
    final_confidence: float
    topology: str
    n_agents: int
    agent_outputs: List[AgentOutput]
    total_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    context_chars: int
    parse_failures: int
    parse_retries: int
    disagreement: bool
    escalated: bool
    ambiguity: Optional[float]
