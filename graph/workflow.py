"""LangGraph workflow for PrivMAS.

This module defines a minimal *hierarchical* multi-agent workflow:

    Supervisor -> (Detector/Analyzer/Validator specialists) -> Supervisor aggregate

Specialists do not communicate with each other; they only log messages to the
supervisor via state updates.

The workflow is intentionally simple and sequential to keep overhead accounting
and ablation studies straightforward.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, TypedDict

from agents.supervisor import Supervisor
from core.state import AgentMessage, MaskingResult, PrivMASState
from privacy.masker import PrivacyMasker


class PrivMASStateDict(TypedDict, total=False):
    run_id: str
    text: str
    label: Optional[str]
    plan: Dict[str, Any]
    specialist_results: Dict[str, Dict[str, Any]]
    final_masked_text: str
    final_strategy: str
    messages: List[Dict[str, Any]]
    timings_ms: Dict[str, float]
    metrics: Dict[str, Any]
    errors: List[str]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _add_message(state: PrivMASStateDict, msg: AgentMessage) -> List[Dict[str, Any]]:
    messages = list(state.get("messages") or [])
    messages.append(msg.model_dump())
    return messages


def _find_task(plan: Optional[Dict[str, Any]], role: str, index: int) -> Optional[Dict[str, Any]]:
    if not plan:
        return None
    tasks = plan.get("tasks") or []
    for t in tasks:
        if (t.get("role") == role) and int(t.get("index", -1)) == index:
            return t
    return None


def build_privmas_graph(config: Optional[Dict[str, Any]] = None):
    """Build and compile the LangGraph workflow."""

    try:
        from langgraph.graph import END, StateGraph
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "langgraph is required for PrivMAS. Install it via requirements.txt"
        ) from e

    cfg = config or {}
    supervisor = Supervisor(cfg)
    masker = PrivacyMasker(cfg)

    roles_cfg = ((cfg.get("agents") or {}).get("roles") or {})

    def role_count(role: str) -> int:
        return int(((roles_cfg.get(role) or {}).get("count") or 0))

    def role_strategy(role: str) -> Optional[str]:
        s = (roles_cfg.get(role) or {}).get("strategy")
        if s in (None, "", "auto"):
            return None
        return str(s)

    # --- Nodes ---

    def supervisor_plan_node(state: PrivMASStateDict) -> Dict[str, Any]:
        text = state.get("text") or ""
        plan = supervisor.plan(text)

        metrics = dict(state.get("metrics") or {})
        if getattr(supervisor, "last_llm_usage", None) is not None or getattr(
            supervisor, "last_llm_usage_approx", None
        ) is not None:
            llm_usage = dict(metrics.get("llm_usage") or {})
            if supervisor.last_llm_usage is not None:
                llm_usage["supervisor_plan"] = supervisor.last_llm_usage
            if supervisor.last_llm_usage_approx is not None:
                llm_usage["supervisor_plan_approx"] = supervisor.last_llm_usage_approx
            metrics["llm_usage"] = llm_usage

        msg = AgentMessage(
            from_agent="supervisor",
            to_agent="supervisor",
            kind="plan",
            content=f"planned_tasks={len(plan.tasks)}",
            content_bytes=len(f"planned_tasks={len(plan.tasks)}".encode("utf-8")),
            ts_ms=_now_ms(),
        )

        return {
            "plan": plan.model_dump(),
            "messages": _add_message(state, msg),
            "metrics": metrics,
        }

    def make_specialist_node(role: str, index: int) -> Callable[[PrivMASStateDict], Dict[str, Any]]:
        agent_id = f"{role}_{index}"
        default_strategy = role_strategy(role)

        def _node(state: PrivMASStateDict) -> Dict[str, Any]:
            text = state.get("text") or ""
            plan = state.get("plan")

            task = _find_task(plan, role, index)
            chosen_strategy = (task or {}).get("strategy") or default_strategy

            result: MaskingResult = masker.mask(
                text,
                agent_id=agent_id,
                role=role,  # type: ignore[arg-type]
                strategy=chosen_strategy,  # type: ignore[arg-type]
            )

            specialist_results = dict(state.get("specialist_results") or {})
            specialist_results[agent_id] = result.model_dump()

            # Specialist -> supervisor message (coordination accounting)
            content = (
                f"role={role} strategy={result.strategy} "
                f"latency_ms={result.latency_ms:.1f} "
                f"placeholders={int((result.details or {}).get('placeholder_count', 0))}"
            )

            msg = AgentMessage(
                from_agent=agent_id,
                to_agent="supervisor",
                kind="result",
                content=content,
                content_bytes=len(content.encode("utf-8")),
                ts_ms=_now_ms(),
            )

            timings_ms = dict(state.get("timings_ms") or {})
            timings_ms[f"{agent_id}_ms"] = result.latency_ms

            errors = list(state.get("errors") or [])
            if result.error:
                errors.append(f"{agent_id}: {result.error}")

            llm_error = (result.details or {}).get("llm_error")
            if llm_error:
                s = str(llm_error)
                if len(s) > 200:
                    s = s[:200] + "..."
                errors.append(f"{agent_id}: llm_error: {s}")

            return {
                "specialist_results": specialist_results,
                "messages": _add_message(state, msg),
                "timings_ms": timings_ms,
                "errors": errors,
            }

        return _node

    def supervisor_aggregate_node(state: PrivMASStateDict) -> Dict[str, Any]:
        # Validate into Pydantic for aggregation convenience
        parsed = PrivMASState.model_validate(state)
        final_text, final_strategy = supervisor.aggregate(parsed)

        msg = AgentMessage(
            from_agent="supervisor",
            to_agent="supervisor",
            kind="aggregate",
            content=f"final_strategy={final_strategy}",
            content_bytes=len(f"final_strategy={final_strategy}".encode("utf-8")),
            ts_ms=_now_ms(),
        )

        return {
            "final_masked_text": final_text,
            "final_strategy": final_strategy,
            "messages": _add_message(state, msg),
        }

    # --- Graph build ---

    g = StateGraph(PrivMASStateDict)
    g.add_node("supervisor_plan", supervisor_plan_node)

    specialist_node_names: List[str] = []
    for role in ("detector", "analyzer", "validator"):
        for i in range(role_count(role)):
            name = f"{role}_{i}"
            g.add_node(name, make_specialist_node(role, i))
            specialist_node_names.append(name)

    g.add_node("supervisor_aggregate", supervisor_aggregate_node)

    g.set_entry_point("supervisor_plan")

    if specialist_node_names:
        g.add_edge("supervisor_plan", specialist_node_names[0])
        for a, b in zip(specialist_node_names, specialist_node_names[1:]):
            g.add_edge(a, b)
        g.add_edge(specialist_node_names[-1], "supervisor_aggregate")
    else:
        g.add_edge("supervisor_plan", "supervisor_aggregate")

    g.add_edge("supervisor_aggregate", END)

    return g.compile()


def run_privmas_once(
    *,
    text: str,
    label: Optional[str] = None,
    run_id: str = "",
    config: Optional[Dict[str, Any]] = None,
) -> PrivMASState:
    """Convenience wrapper: build the graph, run once, return validated state."""

    app = build_privmas_graph(config)

    init = PrivMASState(run_id=run_id, text=text, label=label)

    start = time.perf_counter()
    out_state = app.invoke(init.model_dump())
    e2e_ms = (time.perf_counter() - start) * 1000.0

    # Attach end-to-end timing
    out_state = dict(out_state)
    timings_ms = dict(out_state.get("timings_ms") or {})
    timings_ms["e2e_ms"] = e2e_ms
    out_state["timings_ms"] = timings_ms

    return PrivMASState.model_validate(out_state)
