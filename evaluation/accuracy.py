"""Accuracy evaluation metrics for PII detection."""

from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple


LABEL_ALIASES = {
    "PERSON": "NAME",
    "GPE": "ADDRESS",
    "LOC": "ADDRESS",
    "ORG": "ORG",
}


def normalize_label(label: Any) -> str:
    if label is None:
        return ""
    return LABEL_ALIASES.get(str(label).upper(), str(label).upper())


def entity_to_tuple(entity: Dict[str, Any]) -> Tuple[str, int, int]:
    return (
        normalize_label(entity.get("label", entity.get("entity_type", ""))),
        int(entity.get("start", -1)),
        int(entity.get("end", -1)),
    )

def calculate_iou(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> float:
    """Calculate the Intersection over Union (IoU) of two spans."""
    if span_a[0] >= span_b[0] and span_a[1] <= span_b[1]:
        return 1.0
    if span_a[0] >= span_b[0] and span_a[1] > span_b[1]:
        return (span_a[1] - span_b[0]) / (span_b[1] - span_b[0])
    if span_a[0] < span_b[0] and span_a[1] <= span_b[1]:
        return (span_a[1] - span_b[0]) / (span_b[1] - span_b[0])
    if span_a[0] < span_b[0] and span_a[1] > span_b[1]:
        return 0.0
    return 0.0

def calculate_leakage_rate(tp: int, fn: int) -> float:
    """Calculate privacy leakage as the fraction of ground-truth PII left unmasked."""
    total_ground_truth = tp + fn
    return fn / total_ground_truth if total_ground_truth > 0 else 0.0

def evaluate_pii_detection(text: str, predicted_entities: List[Dict[str, Any]], ground_truth_entities: List[Dict[str, Any]], iou_threshold: float = 0.5) -> Dict:
    """
    Compares predicted PII entities with ground truth entities and computes metrics.

    Args:
        text: The original text.
        predicted_entities: A list of entities predicted by the model.
        ground_truth_entities: A list of ground truth entities.

    Returns:
        A dictionary containing TP, FP, FN counts, and precision, recall, f1 scores.
    """
    
    def to_entity_set(entities: List[Dict[str, Any]]) -> Set[Tuple[str, int, int]]:
        """Converts a list of entity dicts to a set of normalized tuples."""
        return {entity_to_tuple(e) for e in entities}

    predicted_set = to_entity_set(predicted_entities)
    ground_truth_set = to_entity_set(ground_truth_entities)

    # 1. Strict Matching
    strict_tp = len(predicted_set.intersection(ground_truth_set))
    strict_fp = len(predicted_set.difference(ground_truth_set))
    strict_fn = len(ground_truth_set.difference(predicted_set))

    strict_precision = strict_tp / (strict_tp + strict_fp) if (strict_tp + strict_fp) > 0 else 0.0
    strict_recall = strict_tp / (strict_tp + strict_fn) if (strict_tp + strict_fn) > 0 else 0.0
    strict_f1 = 2 * (strict_precision * strict_recall) / (strict_precision + strict_recall) if (strict_precision + strict_recall) > 0 else 0.0

    # 2. Overlap Matching
    overlap_tp = 0
    overlap_fp = 0
    overlap_fn = 0

    for pred_entity in predicted_entities:
        pred_label = pred_entity.get("label", pred_entity.get("entity_type", ""))
        pred_start, pred_end = pred_entity.get("start", -1), pred_entity.get("end", -1)

        for true_entity in ground_truth_entities:
            true_label = true_entity.get("label", true_entity.get("entity_type", ""))
            true_start, true_end = true_entity.get("start", -1), true_entity.get("end", -1)

            if pred_label == true_label:
                iou = calculate_iou((pred_start, pred_end), (true_start, true_end))
                if iou > 0:
                    overlap_tp += 1
            else:
                overlap_fp += 1

    overlap_fp += len(predicted_entities)
    overlap_fn += len(ground_truth_entities)

    overlap_precision = overlap_tp / (overlap_tp + overlap_fp) if (overlap_tp + overlap_fp) > 0 else 0.0
    overlap_recall = overlap_tp / (overlap_tp + overlap_fn) if (overlap_tp + overlap_fn) > 0 else 0.0
    overlap_f1 = 2 * (overlap_precision * overlap_recall) / (overlap_precision + overlap_recall) if (overlap_precision + overlap_recall) > 0 else 0.0

    # 3. Leakage Rate
    strict_leakage = calculate_leakage_rate(strict_tp, strict_fn)
    overlap_leakage = calculate_leakage_rate(overlap_tp, overlap_fn)

    return {
        "strict": {
            "tp": strict_tp, "fp": strict_fp, "fn": strict_fn,
            "precision": strict_precision, "recall": strict_recall, "f1": strict_f1,
            "leakage_rate": strict_leakage,
        },
        "overlap": {
            "tp": overlap_tp, "fp": overlap_fp, "fn": overlap_fn,
            "precision": overlap_precision, "recall": overlap_recall, "f1": overlap_f1,
            "leakage_rate": overlap_leakage,
        },
        "strict_leakage_rate": strict_leakage,
        "overlap_leakage_rate": overlap_leakage,
        # Backward-compatible alias used by some summaries.
        "leakage_rate": overlap_leakage,
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
        if not isinstance(result, dict):
            continue

        pred_entities = result.get('predicted_entities', [])
        true_entities = result.get('ground_truth_entities', [])

        if not isinstance(pred_entities, list):
            pred_entities = []
        if not isinstance(true_entities, list):
            true_entities = []

        pred_set = {entity_to_tuple(e) for e in pred_entities}
        true_set = {entity_to_tuple(e) for e in true_entities}

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
