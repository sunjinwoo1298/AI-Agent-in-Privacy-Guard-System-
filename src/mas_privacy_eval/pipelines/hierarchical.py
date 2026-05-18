"""Hierarchical (supervisor-directed) topology."""

from __future__ import annotations

import numpy as np

from mas_privacy_eval.agents.provider import AgentProvider
from mas_privacy_eval.data.models import PrivacySample
from mas_privacy_eval.pipelines.types import PipelineResult, TopologyType


class HierarchicalPipeline:
    """Supervisor Detector triages; additional agents refine; Consensus resolves."""

    def __init__(self, *, n_agents: int, agent_provider: AgentProvider) -> None:
        self.n_agents = int(n_agents)
        self.topology = TopologyType.HIERARCHICAL
        self._provider = agent_provider

    def run_sample(self, sample: PrivacySample) -> PipelineResult:
        all_outputs = []
        context = ""

        n_detectors = max(1, (self.n_agents - 1) // 3 + 1)
        det_outputs = []
        for i in range(min(n_detectors, self.n_agents)):
            det = self._provider.make_agent(f"hier-det-{i}", "Detector")
            out = det.infer(sample.text, prior_context=context)
            det_outputs.append(out)
            all_outputs.append(out)
            if out.reasoning:
                context += f"\n[Detector-{i}] label={out.label} conf={out.confidence:.2f}: {out.reasoning[:200]}"

        # Level 1: Analyzer
        n_analyzers = max(1, (self.n_agents - len(all_outputs)) // 2)
        for i in range(min(n_analyzers, max(0, self.n_agents - len(all_outputs)))):
            ana = self._provider.make_agent(f"hier-ana-{i}", "Analyzer")
            out = ana.infer(sample.text, prior_context=context)
            all_outputs.append(out)
            if out.reasoning:
                context += f"\n[Analyzer-{i}] label={out.label} conf={out.confidence:.2f}: {out.reasoning[:200]}"

        # Level 2: Validator
        remaining = self.n_agents - len(all_outputs)
        if remaining > 1:
            val = self._provider.make_agent("hier-val-0", "Validator")
            out = val.infer(sample.text, prior_context=context)
            all_outputs.append(out)
            if out.reasoning:
                context += f"\n[Validator] label={out.label} conf={out.confidence:.2f}: {out.reasoning[:200]}"
            remaining -= 1

        # Level 3: Consensus
        if remaining >= 1:
            con = self._provider.make_agent("hier-con-0", "Consensus")
            out = con.infer(sample.text, prior_context=context)
            all_outputs.append(out)

        return self._finalize(sample, all_outputs, context)

    def _finalize(self, sample: PrivacySample, outputs, context: str) -> PipelineResult:
        valid = [o for o in outputs if o.label is not None]
        if not valid:
            final_pred, final_conf = None, 0.5
        else:
            weighted_pos = sum(o.confidence for o in valid if o.label == 1)
            weighted_neg = sum(o.confidence for o in valid if o.label == 0)
            total_conf = sum(o.confidence for o in valid)
            final_pred = 1 if weighted_pos >= weighted_neg else 0
            final_conf = max(weighted_pos, weighted_neg) / (total_conf + 1e-9)

        labels = [o.label for o in valid]
        disagree = len(set(labels)) > 1 if labels else False

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
            escalated=False,
            ambiguity=getattr(sample, "ambiguity", None),
        )
