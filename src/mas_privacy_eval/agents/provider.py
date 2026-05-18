"""Agent provider/factory utilities.

Pipelines should be agnostic to the backend (real HF model vs dry-run heuristic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mas_privacy_eval.agents.hf_agent import RealLLMAgent
from mas_privacy_eval.agents.heuristic_agent import HeuristicPrivacyAgent
from mas_privacy_eval.config import ModelConfig


class AgentLike(Protocol):
    agent_id: str
    role: str

    def infer(self, text: str, prior_context: str = ""):
        ...


class AgentProvider(Protocol):
    def make_agent(self, agent_id: str, role: str) -> AgentLike:
        ...


@dataclass(frozen=True)
class HFRealAgentProvider:
    hf_model: object
    hf_tokenizer: object
    model_cfg: ModelConfig
    verbose: bool = False

    def make_agent(self, agent_id: str, role: str) -> AgentLike:
        return RealLLMAgent(
            agent_id=agent_id,
            role=role,
            hf_model=self.hf_model,
            hf_tokenizer=self.hf_tokenizer,
            model_cfg=self.model_cfg,
            verbose=self.verbose,
        )


@dataclass(frozen=True)
class HeuristicAgentProvider:
    seed: int

    def make_agent(self, agent_id: str, role: str) -> AgentLike:
        return HeuristicPrivacyAgent(agent_id=agent_id, role=role, seed=self.seed)
