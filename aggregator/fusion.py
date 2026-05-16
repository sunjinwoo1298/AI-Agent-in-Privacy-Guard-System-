"""
Entity fusion and aggregation logic for PrivMAS.
"""
from typing import List, Dict, Any

def aggregate_entities(agent_results: List[Any]) -> List[Dict[str, Any]]:
    """
    Aggregates entities from multiple agent results, merging overlapping entities.
    
    This function takes the raw outputs from various agents, extracts all detected
    entities, and then intelligently merges any entities that overlap. This is
    a critical step to prevent the evaluation from penalizing the system for
    redundant detections from multiple agents.

    The merging logic is as follows:
    1. All entities are collected into a single list.
    2. The list is sorted by the starting position of each entity.
    3. The function iterates through the sorted list, merging any entity that
       overlaps with the previously merged one. The span is extended to cover
       both, and for this implementation, the label of the first entity is kept.
    """
    all_entities = []
    
    # The structure of agent_results can vary depending on the workflow.
    # This handles the list of state objects from the parallel run.
    for result_state in agent_results:
        if hasattr(result_state, 'specialist_results') and isinstance(result_state.specialist_results, list):
            for specialist_result in result_state.specialist_results:
                if isinstance(specialist_result, dict) and 'entities' in specialist_result:
                    entities = specialist_result.get('entities', [])
                    if isinstance(entities, list):
                        all_entities.extend(entities)

    if not all_entities:
        return []

    # Ensure start and end are integers for sorting and comparison
    try:
        for e in all_entities:
            e['start'] = int(e.get('start', 0))
            e['end'] = int(e.get('end', 0))
        sorted_entities = sorted(all_entities, key=lambda x: x['start'])
    except (ValueError, TypeError) as e:
        print(f"Error preparing entities for sorting: {e}")
        return [] 

    if not sorted_entities:
        return []

    merged_entities: List[Dict[str, Any]] = [sorted_entities[0]]

    for current_entity in sorted_entities[1:]:
        last_entity = merged_entities[-1]

        # Check for overlap
        if current_entity['start'] < last_entity['end']:
            # --- Fusion Logic ---
            # Simple merge: extend the span to cover both entities.
            # Keep the label of the entity that started first.
            new_end = max(last_entity['end'], current_entity['end'])
            last_entity['end'] = new_end
        else:
            # No overlap, add as a new entity
            merged_entities.append(current_entity)

    return merged_entities
