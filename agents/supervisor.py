"""Supervisor logic for PrivMAS.

The supervisor is responsible for:
1) Task decomposition: create a plan with N specialists per role
2) Routing: choose masking strategies per specialist (heuristics by default)
3) Aggregation: select the final masked text from specialist outputs

Design constraints
------------------
- Keep the system simple and ablation-friendly.
- Specialists do not talk to each other; they only report to the supervisor.
- LLM usage is optional; the default plan is deterministic heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from config import GROQ_API_KEY
from core.state import MaskingStrategy, PrivMASState, SpecialistTask, SupervisorPlan
from evaluation.token_tracking import approx_token_count, normalize_usage


@dataclass(frozen=True)
class RoleConfig:
    count: int
    strategy: Optional[str]


class Supervisor:
    """Minimal supervisor for hierarchical privacy masking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self.last_llm_usage: Optional[Dict[str, Any]] = None
        self.last_llm_usage_approx: Optional[Dict[str, Any]] = None

    @staticmethod
    def _llm_available() -> bool:
        return bool(GROQ_API_KEY) and GROQ_API_KEY != "YOUR_API_KEY"

    def _role_config(self, role: str) -> RoleConfig:
        roles = ((self._config.get("agents") or {}).get("roles") or {})
        role_cfg = roles.get(role) or {}
        count = int(role_cfg.get("count", 1))
        strategy = role_cfg.get("strategy")
        if strategy in (None, "", "auto"):
            strategy = None
        return RoleConfig(count=count, strategy=strategy)

    def _use_llm_planner(self) -> bool:
        sup = self._config.get("supervisor") or {}
        return bool(sup.get("use_llm", False)) and self._llm_available()

    def _supervisor_model(self) -> str:
        sup = self._config.get("supervisor") or {}
        return str(sup.get("model", "llama3-70b-8192"))

    def plan(self, text: str) -> SupervisorPlan:
        """Create a SupervisorPlan.

        Default behavior is a deterministic plan derived from config.
        If enabled, an LLM can return a JSON plan that matches SupervisorPlan.
        """

        # Reset per-call metadata
        self.last_llm_usage = None
        self.last_llm_usage_approx = None

        if self._use_llm_planner():
            plan = self._plan_with_llm(text)
            if plan is not None:
                return plan

        return self._plan_heuristic(text)

    def _plan_heuristic(self, text: str) -> SupervisorPlan:
        det = self._role_config("detector")
        ana = self._role_config("analyzer")
        val = self._role_config("validator")

        tasks: List[SpecialistTask] = []
        for i in range(det.count):
            tasks.append(
                SpecialistTask(
                    role="detector",
                    index=i,
                    strategy=det.strategy,  # often regex_only
                    instructions="Run fast PII masking to catch obvious patterns.",
                )
            )
        for i in range(ana.count):
            tasks.append(
                SpecialistTask(
                    role="analyzer",
                    index=i,
                    strategy=ana.strategy,  # often spacy_plus
                    instructions="Run deeper entity-based masking and refine output.",
                )
            )
        for i in range(val.count):
            tasks.append(
                SpecialistTask(
                    role="validator",
                    index=i,
                    strategy=val.strategy,  # often hybrid_llm
                    instructions="Validate masking quality; use LLM hybrid if enabled.",
                )
            )

        rationale = (
            "Heuristic plan: detectors run fast masking; analyzers add NER; "
            "validators optionally use LLM hybrid to reduce residual PII."
        )

        # If text is extremely short, bias all roles towards regex_only.
        if len(text) < 80:
            for t in tasks:
                t.strategy = "regex_only"  # type: ignore[assignment]
            rationale += " Input is very short; biasing strategies to regex_only."

        return SupervisorPlan(tasks=tasks, rationale=rationale)

    def _plan_with_llm(self, text: str) -> Optional[SupervisorPlan]:
        """Optional structured-output planner.

        Returns None on any failure and falls back to heuristic planning.
        """

        try:
            from groq import Groq
        except Exception:
            return None

        det = self._role_config("detector")
        ana = self._role_config("analyzer")
        val = self._role_config("validator")

        schema_hint = {
            "tasks": [
                {
                    "role": "detector",
                    "index": 0,
                    "strategy": "regex_only",
                    "instructions": "...",
                }
            ],
            "rationale": "...",
        }

        prompt = (
            "You are a supervisor for a privacy masking multi-agent system. "
            "Return ONLY valid JSON (no markdown) matching this schema: "
            f"{json.dumps(schema_hint)}\n\n"
            "Allowed strategies: regex_only, spacy_plus, hybrid_fast, llm_only, hybrid_llm.\n"
            f"Agent counts: detector={det.count}, analyzer={ana.count}, validator={val.count}.\n"
            "Create exactly that many tasks.\n\n"
            f"Text: {text!r}"
        )

        try:
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self._supervisor_model(),
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""

            # Token usage tracking (exact when provider returns usage, plus a stable approximation)
            self.last_llm_usage = normalize_usage(getattr(resp, "usage", None))
            self.last_llm_usage_approx = {
                "prompt_tokens": approx_token_count(prompt),
                "completion_tokens": approx_token_count(raw),
                "total_tokens": approx_token_count(prompt) + approx_token_count(raw),
                "model": self._supervisor_model(),
            }

            # Extract first JSON object defensively
            match = re.search(r"\{[\s\S]*\}", raw)
            if not match:
                return None
            data = json.loads(match.group(0))
            return SupervisorPlan.model_validate(data)

        except Exception:
            return None

    def aggregate(self, state: PrivMASState) -> Tuple[str, MaskingStrategy]:
        """Aggregate specialist outputs into a final result."""

        policy = ((self._config.get("aggregation") or {}).get("policy") or "prefer_validator")

        results = list(state.specialist_results.values())
        if not results:
            return state.text, "regex_only"

        def is_ok(r) -> bool:
            return r.error is None

        def by_role(role: str) -> List[Any]:
            return [r for r in results if r.role == role and is_ok(r)]

        if policy == "prefer_validator":
            pick = (by_role("validator") or by_role("analyzer") or by_role("detector"))
            if pick:
                r0 = pick[0]
                return r0.masked_text, r0.strategy

        if policy == "prefer_analyzer":
            pick = (by_role("analyzer") or by_role("validator") or by_role("detector"))
            if pick:
                r0 = pick[0]
                return r0.masked_text, r0.strategy

        if policy == "prefer_fastest":
            ok = [r for r in results if is_ok(r)] or results
            r0 = min(ok, key=lambda r: r.latency_ms)
            return r0.masked_text, r0.strategy

        if policy == "prefer_most_masked":
            def placeholder_count(r) -> int:
                return int((r.details or {}).get("placeholder_count", 0))

            ok = [r for r in results if is_ok(r)] or results
            r0 = max(ok, key=placeholder_count)
            return r0.masked_text, r0.strategy

        # Default fallback
        r0 = next((r for r in results if is_ok(r)), results[0])
        return r0.masked_text, r0.strategy
