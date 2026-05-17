"""Rule-based heuristic for assigning masking strategies to agents."""

from typing import List
from core.state import MaskingStrategy

def assign_strategies(num_agents: int) -> List[MaskingStrategy]:
    """
    Assigns masking strategies to a given number of agents based on a heuristic.

    Heuristic:
    1. Use the shared GLiNER-PII strategy for every agent.

    Args:
        num_agents: The total number of agents.

    Returns:
        A list of masking strategies, one for each agent.
    """
    strategies: List[MaskingStrategy] = []
    
    if num_agents == 0:
        return []

    for _ in range(num_agents):
        strategies.append("gliner_pii")

    return strategies
