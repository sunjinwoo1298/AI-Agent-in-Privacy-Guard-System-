"""Shared state schemas for PrivMAS.

This module defines the Pydantic models used across the multi-agent workflow.
It intentionally contains *no* heavy imports (spaCy / Groq / LangGraph) so it can
be imported safely in any context (including unit tests).

Key idea
--------
PrivMAS is evaluated by running the *same* masking task under different agent
counts/topologies. The state captures:
- Input text and optional dataset label
- Supervisor plan
- Per-specialist outputs + timings
- Communication messages (for overhead metrics)
- Final aggregated masking result
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


MaskingStrategy = Literal[
    "regex_only",
    "spacy_plus",
    "parallel_generalist",
]

RoleName = Literal["detector", "analyzer", "validator", "supervisor", "generalist"]


class AgentMessage(BaseModel):
    """A lightweight message record used to quantify coordination overhead."""

    from_agent: str
    to_agent: str
    kind: str = "info"
    content: str = ""
    content_bytes: int = 0
    ts_ms: Optional[int] = None


class SpecialistTask(BaseModel):
    """A unit of work created by the supervisor for a specialist agent."""

    role: RoleName
    index: int = 0
    strategy: Optional[MaskingStrategy] = None
    instructions: str = ""

    @property
    def agent_id(self) -> str:
        return f"{self.role}_{self.index}"


class SupervisorPlan(BaseModel):
    """Supervisor decomposition of a masking job."""

    tasks: List[SpecialistTask] = Field(default_factory=list)
    rationale: str = ""


class MaskingResult(BaseModel):
    """Output of a masking attempt by a specialist or the supervisor."""

    agent_id: str
    role: RoleName
    strategy: MaskingStrategy
    masked_text: str
    latency_ms: float = 0.0
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class PrivMASState(BaseModel):
    """Top-level state passed through the PrivMAS workflow."""

    # Input
    run_id: str = ""
    text: str
    label: Optional[str] = None

    # Planning / routing
    plan: Optional[SupervisorPlan] = None

    # Specialist outputs (can be a dict for sequential or list for parallel)
    specialist_results: Dict[str, MaskingResult] | List[Dict[str, Any]] = Field(default_factory=dict)

    # Supervisor aggregation
    final_masked_text: Optional[str] = None
    final_strategy: Optional[MaskingStrategy] = None

    # Coordination overhead
    messages: List[AgentMessage] = Field(default_factory=list)

    # Timings / metrics
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
