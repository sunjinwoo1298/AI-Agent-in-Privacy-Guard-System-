"""Orchestrator: split text into deterministic shards, dispatch agents, aggregate results."""

import time
import concurrent.futures
import logging
import uuid
from typing import List, Dict, Any

from agents.deterministic_agent import DeterministicAgent, AgentResult
from utils.sharding import split_text_into_sentences
from aggregator.aggregator import merge_masked_sentences


def _process_shard_in_worker(agent_id, shard, use_spacy_flag: bool):
    """Top-level worker function used with ProcessPoolExecutor.

    Instantiates a local DeterministicAgent and processes the shard.
    """
    # import locally to avoid issues in worker pickling
    from agents.deterministic_agent import DeterministicAgent

    local_agent = DeterministicAgent(agent_id, nlp=None)
    if use_spacy_flag:
        try:
            import spacy

            local_agent.nlp = spacy.load("en_core_web_sm")
        except Exception:
            local_agent.nlp = None
    return local_agent.process(shard)


def run_pipeline(text: str, n_agents: int = 2, nlp=None, use_spacy: bool = False, use_process_pool: bool = False):
    """Run the pipeline on `text` using `n_agents` agents.

    Returns: (final_masked_text, metrics_dict, agent_results_list)
    """
    n_agents = max(1, int(n_agents))

    # Optional: use preloaded nlp if provided, else load if use_spacy True
    if nlp is None and use_spacy:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logging.warning("spaCy load failed, continuing without it: %s", e)
            nlp = None

    # Deterministic sentence splitting with absolute offsets
    sentences = split_text_into_sentences(text)

    # Build deterministic round-robin shards: sentence index -> agent (index % n)
    shards: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_agents)}
    for s in sentences:
        agent_id = s["index"] % n_agents
        shards[agent_id].append(s)

    # Dispatch
    dispatch_start = time.perf_counter()
    agent_results: List[AgentResult] = []

    if use_process_pool:
        # Use a ProcessPoolExecutor: instantiate agents inside each worker to avoid pickling heavy models
        def _process_shard_worker(args):
            # args: (agent_id, shard, use_spacy)
            agent_id, shard, use_spacy_flag = args
            # instantiate agent locally in worker
            local_agent = DeterministicAgent(agent_id, nlp=None)
            if use_spacy_flag:
                try:
                    import spacy

                    local_agent.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    local_agent.nlp = None
            return local_agent.process(shard)

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_agents) as executor:
            futures = {executor.submit(_process_shard_in_worker, i, shards[i], use_spacy): i for i in range(n_agents)}
            dispatch_end = time.perf_counter()
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                # arrival time recorded in parent clock
                res.arrival_time = time.perf_counter()
                agent_results.append(res)

    else:
        # ThreadPool path (default)
        agents = {i: DeterministicAgent(i, nlp=nlp) for i in range(n_agents)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_agents) as executor:
            future_to_agent = {}
            for i in range(n_agents):
                future = executor.submit(agents[i].process, shards[i])
                future_to_agent[future] = i

            dispatch_end = time.perf_counter()

            # Collect results as they arrive
            for future in concurrent.futures.as_completed(future_to_agent):
                res = future.result()
                # record arrival time (synchronization observation)
                res.arrival_time = time.perf_counter()
                agent_results.append(res)
    # Aggregation
    aggregation_start = time.perf_counter()
    final_text = merge_masked_sentences(agent_results)
    aggregation_end = time.perf_counter()

    # Compute metrics
    total_latency = aggregation_end - dispatch_start
    dispatch_duration = dispatch_end - dispatch_start
    aggregation_duration = aggregation_end - aggregation_start
    agent_exec_times = [r.processing_time for r in agent_results]
    arrival_times = [r.arrival_time for r in agent_results]
    critical_path = max(agent_exec_times) if agent_exec_times else 0.0
    sync_delay = (max(arrival_times) - min(arrival_times)) if arrival_times else 0.0
    coordination_tax = total_latency - critical_path
    tokens_total = sum(r.tokens_processed for r in agent_results)
    per_agent_throughput = [
        (r.tokens_processed / r.processing_time) if r.processing_time > 0 else 0.0
        for r in agent_results
    ]
    total_throughput = (tokens_total / total_latency) if total_latency > 0 else 0.0

    metrics = {
        "run_id": str(uuid.uuid4()),
        "n_agents": n_agents,
        "total_latency": total_latency,
        "dispatch_duration": dispatch_duration,
        "aggregation_duration": aggregation_duration,
        "critical_path": critical_path,
        "sync_delay": sync_delay,
        "coordination_tax": coordination_tax,
        "efficiency": (critical_path / total_latency) if total_latency > 0 else 0.0,
        "tokens_total": tokens_total,
        "per_agent_throughput": per_agent_throughput,
        "total_throughput": total_throughput,
        "agent_exec_times": agent_exec_times,
    }

    return final_text, metrics, agent_results


__all__ = ["run_pipeline"]
