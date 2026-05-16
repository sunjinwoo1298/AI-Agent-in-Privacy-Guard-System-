"""Parallel workflow for PrivMAS using data sharding and threading."""

import queue
import threading
import time
from typing import Any, Dict, List, Optional

from agents.generalist import GeneralistAgent
from agents.strategy_assigner import assign_strategies
from core.state import PrivMASState, MaskingStrategy
from utils.sharding import shard_text


def _normalize_entity(entity: Dict[str, Any], chunk_offset: int) -> Dict[str, Any]:
    """Convert chunk-local entity offsets into global text offsets."""

    label = entity.get("label", entity.get("entity_type", ""))
    start = entity.get("start")
    end = entity.get("end")

    normalized = dict(entity)
    normalized["label"] = label

    if isinstance(start, int):
        normalized["start"] = start + chunk_offset
    if isinstance(end, int):
        normalized["end"] = end + chunk_offset

    return normalized


def generalist_worker(
    chunk_queue: queue.Queue,
    result_queue: queue.Queue,
    config: Optional[Dict[str, Any]] = None,
):
    """Worker function for each thread."""
    while not chunk_queue.empty():
        try:
            chunk_id, chunk_text, chunk_offset, strategy = chunk_queue.get_nowait()
            # Each agent is created per task in this model, which is less efficient
            # but ensures statelessness if the agent design requires it.
            # For a more optimized approach, a pool of pre-initialized agents
            # could be used.
            agent = GeneralistAgent(config, assigned_strategy=strategy)
            result = agent.process_chunk(chunk_text, chunk_id)
            result["arrival_time"] = time.perf_counter() # Record arrival time
            result["chunk_offset"] = chunk_offset
            result_queue.put(result)
            chunk_queue.task_done()
        except queue.Empty:
            break
        except Exception as e:
            result = {"chunk_id": -1, "error": str(e), "arrival_time": time.perf_counter(), "chunk_offset": 0}
            result_queue.put(result)
            chunk_queue.task_done()


def run_in_parallel(
    *,
    text: str,
    label: Optional[str] = None,
    run_id: str = "",
    config: Optional[Dict[str, Any]] = None,
) -> PrivMASState:
    """
    Orchestrates the parallel processing of a text for PII masking.
    """
    cfg = config or {}
    num_agents = (cfg.get("agents") or {}).get("generalist_count", 4)
    
    start_time = time.perf_counter()

    # 1. Shard the text and assign strategies
    shards = shard_text(text, num_shards=num_agents)
    strategies = assign_strategies(num_agents)
    
    chunk_queue = queue.Queue()
    chunk_offset = 0
    for i, (shard, strategy) in enumerate(zip(shards, strategies)):
        chunk_queue.put((i, shard, chunk_offset, strategy))
        chunk_offset += len(shard)
        
    result_queue = queue.Queue()
    threads = []

    # 2. Create and start threads
    for _ in range(num_agents):
        thread = threading.Thread(
            target=generalist_worker, args=(chunk_queue, result_queue, cfg)
        )
        threads.append(thread)
        thread.start()

    # 3. Wait for all chunks to be processed
    chunk_queue.join()

    # 4. Wait for all threads to complete
    for thread in threads:
        thread.join()

    # 5. Aggregate results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    e2e_ms = (time.perf_counter() - start_time) * 1000.0

    # Sort results by chunk_id to reassemble in order
    results.sort(key=lambda r: r["chunk_id"])

    errors = [r["error"] for r in results if r.get("error")]
    masked_text = "".join([r["masked_text"] for r in results if "masked_text" in r and not r.get("error")])
    
    # Aggregate detected entities
    all_detected_entities = []
    for r in results:
        if "detected_entities" in r:
            chunk_offset = int(r.get("chunk_offset", 0))
            for entity in r["detected_entities"]:
                if isinstance(entity, dict):
                    all_detected_entities.append(_normalize_entity(entity, chunk_offset))

    # 6. Calculate new metrics
    timings_ms = {"e2e_ms": e2e_ms}
    agent_details = []
    if results:
        # T_inf (Critical Path) - The longest time any single agent took to complete its task.
        t_inf_ms = max(r.get("latency_ms", 0) for r in results)
        timings_ms["t_inf_ms"] = t_inf_ms

        # Δ_sync (Synchronization Delay) - The time difference between the first and last agent finishing.
        arrival_times = [r["arrival_time"] for r in results]
        delta_sync_ms = (max(arrival_times) - min(arrival_times)) * 1000.0 if arrival_times else 0.0
        timings_ms["delta_sync_ms"] = delta_sync_ms

        # C_tax (Coordination Tax) - The overhead of threading, queueing, and synchronization.
        c_tax_ms = e2e_ms - t_inf_ms
        timings_ms["c_tax_ms"] = c_tax_ms
        
        # Capture details for each agent
        for r in results:
            agent_details.append({
                "chunk_id": r.get("chunk_id"),
                "strategy": r.get("strategy"),
                "latency_ms": r.get("latency_ms"),
            })

    # Create a final state object similar to the sequential workflow
    final_state = PrivMASState(
        run_id=run_id,
        text=text,
        label=label,
        final_masked_text=masked_text,
        final_strategy="parallel_generalist",
        aggregated_entities=all_detected_entities,
        timings_ms=timings_ms,
        errors=errors,
        # Other fields can be populated if needed
        plan={},
        specialist_results=agent_details, # Store agent details here
        messages=[],
        metrics={},
    )

    return final_state
