"""Rule-based heuristic for assigning masking strategies to agents."""

from typing import List
from core.state import MaskingStrategy

def assign_strategies(num_agents: int) -> List[MaskingStrategy]:
    """
    Assigns masking strategies to a given number of agents based on a heuristic.

    Heuristic:
    1. Prioritize at least one of each type: regex, spacy.
    2. Distribute remaining agents evenly, starting with cheaper methods.

    Args:
        num_agents: The total number of agents.

    Returns:
        A list of masking strategies, one for each agent.
    """
    strategies: List[MaskingStrategy] = []
    
    if num_agents == 0:
        return []

    # Base strategies to ensure variety
    base_strategies: List[MaskingStrategy] = ["regex_only", "spacy_plus"]

    # Assign strategies
    for i in range(num_agents):
        strategy = base_strategies[i % len(base_strategies)]
        strategies.append(strategy)
            
    return strategies
