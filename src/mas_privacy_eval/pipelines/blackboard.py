"""Blackboard (shared memory) topology."""

from __future__ import annotations

import numpy as np

from mas_privacy_eval.agents.provider import AgentProvider
from mas_privacy_eval.data.models import PrivacySample
from mas_privacy_eval.pipelines.types import PipelineResult, TopologyType


class BlackboardPipeline:
    """Agents read/write to a shared blackboard state."""

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
        self.topology = TopologyType.BLACKBOARD
        self._agents = [
            agent_provider.make_agent(f"bb-{i}", self.ROLE_SEQUENCE[i % len(self.ROLE_SEQUENCE)])
            for i in range(self.n_agents)
        ]

    def run_sample(self, sample: PrivacySample) -> PipelineResult:
        blackboard = {
            "text": sample.text,
            "current_prob": None,  # probability of label=1
            "entries": [],
        }
        all_outputs = []
        alpha = 0.6

        for agent in self._agents:
            if blackboard["entries"]:
                p = blackboard["current_prob"]
                prob_str = "None" if p is None else f"{p:.2f}"
                bb_context = "SHARED BLACKBOARD STATE:\n"
                bb_context += f"Current P(label=1)={prob_str}\nPrior contributions:\n"
                for entry in blackboard["entries"][-4:]:
                    bb_context += f"  [{entry['role']}]: label={entry['label']} | {entry['reasoning'][:150]}\n"
            else:
                bb_context = ""

            out = agent.infer(sample.text, prior_context=bb_context)
            all_outputs.append(out)

            if out.label is None:
                continue

            blackboard["entries"].append(
                {
                    "role": out.role,
                    "label": out.label,
                    "confidence": out.confidence,
                    "reasoning": out.reasoning[:200],
                }
            )

            # Convert (label, confidence) to probability of label=1
            p_new = out.confidence if out.label == 1 else (1.0 - out.confidence)
            if blackboard["current_prob"] is None:
                blackboard["current_prob"] = float(p_new)
            else:
                blackboard["current_prob"] = float(alpha * p_new + (1 - alpha) * blackboard["current_prob"])

        current_prob = blackboard["current_prob"]
        if current_prob is None:
            final_pred, final_conf = None, 0.5
        else:
            final_pred = 1 if current_prob >= 0.5 else 0
            final_conf = float(max(current_prob, 1.0 - current_prob))

        context_str = str(blackboard)
        labels = [o.label for o in all_outputs if o.label is not None]
        disagree = len(set(labels)) > 1 if labels else False

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
            context_chars=len(context_str),
            parse_failures=sum(1 for o in all_outputs if not o.parse_success),
            parse_retries=sum(o.parse_retries for o in all_outputs),
            disagreement=disagree,
            escalated=False,
            ambiguity=getattr(sample, "ambiguity", None),
        )
