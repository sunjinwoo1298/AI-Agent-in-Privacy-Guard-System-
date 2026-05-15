"""Accuracy evaluation metrics for PII detection."""

from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple

def evaluate_pii_detection(predicted_entities: List[Dict[str, Any]], ground_truth_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares predicted PII entities with ground truth entities and computes metrics.

    Args:
        predicted_entities: A list of entities predicted by the model.
        ground_truth_entities: A list of ground truth entities.

    Returns:
        A dictionary containing TP, FP, FN counts, and precision, recall, f1 scores.
    """
    
    def to_entity_set(entities: List[Dict[str, Any]]) -> Set[Tuple[str, int, int]]:
        """Converts a list of entity dicts to a set of tuples for easy comparison."""
        return {(e['label'], e['start'], e['end']) for e in entities}

    predicted_set = to_entity_set(predicted_entities)
    ground_truth_set = to_entity_set(ground_truth_entities)

    tp = len(predicted_set.intersection(ground_truth_set))
    fp = len(predicted_set.difference(ground_truth_set))
    fn = len(ground_truth_set.difference(predicted_set))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_overall_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates overall and per-label metrics from a list of individual results.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for result in all_results:
        pred_entities = result.get('predicted_entities', [])
        true_entities = result.get('ground_truth_entities', [])

        pred_set = {(e['label'], e['start'], e['end']) for e in pred_entities}
        true_set = {(e['label'], e['start'], e['end']) for e in true_entities}

        # Overall stats
        total_tp += len(pred_set.intersection(true_set))
        total_fp += len(pred_set.difference(true_set))
        total_fn += len(true_set.difference(pred_set))

        # Per-label stats
        labels = {e[0] for e in pred_set.union(true_set)}
        for label in labels:
            pred_label_set = {e for e in pred_set if e[0] == label}
            true_label_set = {e for e in true_set if e[0] == label}
            per_label_stats[label]["tp"] += len(pred_label_set.intersection(true_label_set))
            per_label_stats[label]["fp"] += len(pred_label_set.difference(true_label_set))
            per_label_stats[label]["fn"] += len(true_label_set.difference(pred_label_set))

    # Calculate overall scores
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    # Calculate per-label scores
    per_label_metrics = {}
    for label, stats in per_label_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        per_label_metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    return {
        "overall": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn
        },
        "per_label": per_label_metrics
    }
