"""Accuracy evaluation metrics for PII detection."""

from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple, Optional


LABEL_ALIASES = {
    "NAME": "NAME",
    "PER": "NAME",
    "PERSON": "NAME",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "GPE": "ADDRESS",
    "LOC": "ADDRESS",
    "ADDRESS": "ADDRESS",
    "LOCATION": "ADDRESS",
    "EMAIL_ADDRESS": "EMAIL",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "MOBILE_PHONE_NUMBER": "PHONE",
    "LANDLINE_PHONE_NUMBER": "PHONE",
    "FAX_NUMBER": "PHONE",
    "BANK_ACCOUNT": "BANK_ACCOUNT",
    "BANK_ACCOUNT_NUMBER": "BANK_ACCOUNT",
    "IBAN": "BANK_ACCOUNT",
    "SWIFT": "BANK_ACCOUNT",
    "CREDIT_CARD": "CREDIT_CARD",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "SOCIAL_SECURITY_NUMBER": "SSN",
    "SSN": "SSN",
    "PASSPORT": "PASSPORT",
    "PASSPORT_NUMBER": "PASSPORT",
    "DRIVING_LICENSE": "DRIVING_LICENSE",
    "DRIVER_LICENSE": "DRIVING_LICENSE",
    "DRIVER'S_LICENSE_NUMBER": "DRIVING_LICENSE",
    "AADHAR": "AADHAR",
    "AADHAAR": "AADHAR",
    "PAN": "PAN",
    "IP_ADDRESS": "IP_ADDRESS",
    "USERNAME": "USERNAME",
    "PASSWORD": "PASSWORD",
    "API_KEY": "API_KEY",
    "DEVICE_ID": "DEVICE_ID",
    "CRYPTO_WALLET": "CRYPTO_WALLET",
    "MEDICAL_ID": "MEDICAL_ID",
    "VOTER_ID": "VOTER_ID",
    "DOB": "DOB",
    "DATE_OF_BIRTH": "DOB",
    "MISC": "MISC",
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
    """Calculate the Intersection over Union (IoU) of two half-open spans."""

    a_start, a_end = span_a
    b_start, b_end = span_b

    if a_end <= a_start or b_end <= b_start:
        return 0.0

    intersection_start = max(a_start, b_start)
    intersection_end = min(a_end, b_end)
    intersection = max(0, intersection_end - intersection_start)
    if intersection <= 0:
        return 0.0

    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union if union > 0 else 0.0


def _normalize_entity(entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(entity, dict):
        return None

    label = normalize_label(entity.get("label", entity.get("entity_type", "")))
    try:
        start = int(entity.get("start", -1))
        end = int(entity.get("end", -1))
    except (TypeError, ValueError):
        return None

    if start < 0 or end < 0 or end <= start:
        return None

    return {
        "label": label,
        "start": start,
        "end": end,
        "text": entity.get("text", entity.get("value", "")),
    }


def _dedupe_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for entity in entities:
        normalized = _normalize_entity(entity)
        if not normalized:
            continue
        key = (normalized["label"], normalized["start"], normalized["end"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _one_to_one_overlap_match(
    predicted_entities: List[Dict[str, Any]],
    ground_truth_entities: List[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[int, int, int]:
    """Match entities greedily by label and span overlap."""

    preds = _dedupe_entities(predicted_entities)
    truths = _dedupe_entities(ground_truth_entities)

    matched_truth_indices: Set[int] = set()
    tp = 0
    fp = 0

    for pred in preds:
        best_idx = None
        best_score = 0.0

        for idx, true in enumerate(truths):
            if idx in matched_truth_indices:
                continue
            if pred["label"] != true["label"]:
                continue

            score = calculate_iou((pred["start"], pred["end"]), (true["start"], true["end"]))
            if score >= iou_threshold and score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            matched_truth_indices.add(best_idx)
            tp += 1
        else:
            fp += 1

    fn = len(truths) - len(matched_truth_indices)
    return tp, fp, fn

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
        return {entity_to_tuple(e) for e in entities if _normalize_entity(e)}

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
    overlap_tp, overlap_fp, overlap_fn = _one_to_one_overlap_match(
        predicted_entities,
        ground_truth_entities,
        iou_threshold=iou_threshold,
    )

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
