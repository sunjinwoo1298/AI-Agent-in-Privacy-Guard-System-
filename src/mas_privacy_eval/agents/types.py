"""Agent output types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AgentOutput:
    """Structured output from a single agent invocation."""

    agent_id: str
    role: str
    raw_response: str
    parsed: Optional[Dict]
    label: Optional[int]
    confidence: float
    reasoning: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    parse_success: bool
    parse_retries: int
    error_message: Optional[str]
    timestamp: float


class AgentParseError(Exception):
    """Raised when an agent response cannot be parsed as JSON."""
