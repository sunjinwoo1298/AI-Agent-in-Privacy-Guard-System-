"""Metrics utilities: IoU-based span matching and precision/recall/F1."""

from typing import List, Dict, Tuple


def span_iou(pred: Tuple[int, int], gt: Tuple[int, int]) -> float:
    p0, p1 = pred
    g0, g1 = gt
    inter0 = max(p0, g0)
    inter1 = min(p1, g1)
    inter = max(0, inter1 - inter0)
    union = (p1 - p0) + (g1 - g0) - inter
    return (inter / union) if union > 0 else 0.0


def match_predictions_to_ground_truth(preds: List[Dict], gts: List[Dict], iou_threshold: float = 0.5) -> Tuple[int, int, int]:
    """Greedy matching by IoU (one-to-one). Returns (TP, FP, FN).

    preds/gts are dicts with keys: `label`, `start`, `end`.
    Matches require same `label` and IoU >= iou_threshold.
    """
    preds = [p for p in preds]
    gts = [g for g in gts]
    matched_gt = set()
    tp = 0

    for p in preds:
        best_iou = 0.0
        best_j = None
        for j, g in enumerate(gts):
            if j in matched_gt:
                continue
            if p.get("label") != g.get("label"):
                continue
            iou = span_iou((p.get("start"), p.get("end")), (g.get("start"), g.get("end")))
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None:
            matched_gt.add(best_j)
            tp += 1

    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def evaluate_run(agent_results: List[Dict], ground_truth_entities: List[Dict], iou_threshold: float = 0.5) -> Dict:
    """Aggregate predicted entities from agent_results and evaluate against ground truth.

    agent_results: list of AgentResult objects (with `.predicted_entities` list)
    ground_truth_entities: list of dicts with `label`, `start`, `end`.
    Returns dict with TP/FP/FN/precision/recall/f1 and counts.
    """
    preds = []
    for ar in agent_results:
        for p in getattr(ar, "predicted_entities", []):
            # accept both `start`/`end` keys
            if "start" in p and "end" in p:
                preds.append({"label": p.get("label"), "start": p.get("start"), "end": p.get("end")})
            elif "abs_start" in p and "abs_end" in p:
                preds.append({"label": p.get("label"), "start": p.get("abs_start"), "end": p.get("abs_end")})

    gts = [{"label": g.get("label"), "start": g.get("start"), "end": g.get("end")} for g in ground_truth_entities]

    tp, fp, fn = match_predictions_to_ground_truth(preds, gts, iou_threshold=iou_threshold)
    scores = precision_recall_f1(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": len(preds),
        "gt_count": len(gts),
        **scores,
    }


__all__ = ["span_iou", "match_predictions_to_ground_truth", "precision_recall_f1", "evaluate_run"]
