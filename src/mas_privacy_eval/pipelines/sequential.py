"""Sequential chain topology."""

from __future__ import annotations

from typing import List

import numpy as np

from mas_privacy_eval.agents.provider import AgentProvider
from mas_privacy_eval.data.models import PrivacySample
from mas_privacy_eval.pipelines.types import PipelineResult, TopologyType


class SequentialPipeline:
    """Detector → Analyzer → Validator → (optional Consensus), repeating.

Each agent sees the full accumulated reasoning chain.
"""

    ROLE_SEQUENCE = [
        "Detector",
        "Analyzer",
        "Validator",
        "Consensus",
        "Detector",
        "Analyzer",
        "Validator",
        "Consensus",
    ]

    def __init__(self, *, n_agents: int, agent_provider: AgentProvider) -> None:
        self.n_agents = int(n_agents)
        self.topology = TopologyType.SEQUENTIAL

        self._agents = []
        for i in range(self.n_agents):
            role = self.ROLE_SEQUENCE[i % len(self.ROLE_SEQUENCE)]
            self._agents.append(agent_provider.make_agent(f"seq-{i}", role))

    def run_sample(self, sample: PrivacySample) -> PipelineResult:
        context = ""
        outputs = []
        for agent in self._agents:
            out = agent.infer(sample.text, prior_context=context)
            outputs.append(out)
            if out.reasoning:
                context += f"\n[{out.role}] label={out.label} conf={out.confidence:.2f}: {out.reasoning[:300]}"
        return self._finalize(sample, outputs, context)

    def _finalize(self, sample: PrivacySample, outputs, context: str) -> PipelineResult:
        valid = [o for o in outputs if o.label is not None]
        if not valid:
            final_pred, final_conf = None, 0.5
            disagree, escalate = False, True
        else:
            weighted_pos = sum(o.confidence for o in valid if o.label == 1)
            weighted_neg = sum(o.confidence for o in valid if o.label == 0)
            total_conf = sum(o.confidence for o in valid)
            final_pred = 1 if weighted_pos >= weighted_neg else 0
            final_conf = max(weighted_pos, weighted_neg) / (total_conf + 1e-9)
            labels = [o.label for o in valid]
            disagree = len(set(labels)) > 1
            escalate = any(o.parsed and o.parsed.get("escalate", False) for o in outputs)

        return PipelineResult(
            sample_id=sample.sample_id,
            true_label=sample.true_label,
            final_prediction=final_pred,
            final_confidence=float(np.clip(final_conf, 0, 1)),
            topology=self.topology.value,
            n_agents=len(outputs),
            agent_outputs=outputs,
            total_latency_ms=sum(o.latency_ms for o in outputs),
            total_input_tokens=sum(o.input_tokens for o in outputs),
            total_output_tokens=sum(o.output_tokens for o in outputs),
            total_tokens=sum(o.total_tokens for o in outputs),
            context_chars=len(context),
            parse_failures=sum(1 for o in outputs if not o.parse_success),
            parse_retries=sum(o.parse_retries for o in outputs),
            disagreement=disagree,
            escalated=escalate,
            ambiguity=getattr(sample, "ambiguity", None),
        )
