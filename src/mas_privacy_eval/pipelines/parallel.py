"""Parallel + Consensus topology."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from mas_privacy_eval.agents.provider import AgentProvider
from mas_privacy_eval.data.models import PrivacySample
from mas_privacy_eval.pipelines.types import PipelineResult, TopologyType


class ParallelPipeline:
    """N-1 workers run independently, then a Consensus agent aggregates.

If N==1, we run exactly one worker (no Consensus) to match the sweep parameter.
"""

    WORKER_ROLES = ["Detector", "Analyzer", "Validator"]

    def __init__(self, *, n_agents: int, agent_provider: AgentProvider) -> None:
        self.n_agents = int(n_agents)
        self.topology = TopologyType.PARALLEL

        if self.n_agents <= 1:
            self._workers = [agent_provider.make_agent("par-0", self.WORKER_ROLES[0])]
            self._consensus = None
        else:
            n_workers = self.n_agents - 1
            self._workers = [
                agent_provider.make_agent(f"par-{i}", self.WORKER_ROLES[i % len(self.WORKER_ROLES)])
                for i in range(n_workers)
            ]
            self._consensus = agent_provider.make_agent("par-consensus", "Consensus")

    def run_sample(self, sample: PrivacySample) -> PipelineResult:
        worker_outputs = []
        for agent in self._workers:
            worker_outputs.append(agent.infer(sample.text, prior_context=""))

        consensus_context = ""
        consensus_out = None
        all_outputs: List = list(worker_outputs)

        if self._consensus is not None:
            summary_lines = []
            for out in worker_outputs:
                summary_lines.append(
                    f"[{out.role}] label={out.label}, confidence={out.confidence:.2f}, reasoning: {out.reasoning[:200]}"
                )
            consensus_context = "\n".join(summary_lines)

            consensus_out = self._consensus.infer(sample.text, prior_context=consensus_context)
            all_outputs = worker_outputs + [consensus_out]

        # Final decision comes from Consensus agent preferentially
        if consensus_out is not None and consensus_out.label is not None:
            final_pred = consensus_out.label
            final_conf = consensus_out.confidence
        else:
            valid = [o for o in worker_outputs if o.label is not None]
            if valid:
                final_pred = int(round(float(np.mean([o.label for o in valid]))))
                final_conf = float(np.mean([o.confidence for o in valid]))
            else:
                final_pred, final_conf = None, 0.5

        labels = [o.label for o in all_outputs if o.label is not None]
        disagree = len(set(labels)) > 1 if labels else False
        escalate = False
        if consensus_out is not None and consensus_out.parsed:
            escalate = bool(consensus_out.parsed.get("escalate", False))

        return PipelineResult(
            sample_id=sample.sample_id,
            true_label=sample.true_label,
            final_prediction=final_pred,
            final_confidence=float(np.clip(final_conf, 0, 1)),
            topology=self.topology.value,
            n_agents=len(all_outputs),
            agent_outputs=all_outputs,
            total_latency_ms=sum(o.latency_ms for o in all_outputs),
            total_input_tokens=sum(o.input_tokens for o in all_outputs),
            total_output_tokens=sum(o.output_tokens for o in all_outputs),
            total_tokens=sum(o.total_tokens for o in all_outputs),
            context_chars=len(consensus_context),
            parse_failures=sum(1 for o in all_outputs if not o.parse_success),
            parse_retries=sum(o.parse_retries for o in all_outputs),
            disagreement=disagree,
            escalated=escalate,
            ambiguity=getattr(sample, "ambiguity", None),
        )
