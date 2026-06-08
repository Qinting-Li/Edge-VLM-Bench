from __future__ import annotations

from typing import Any


def evaluate_predictions(dataset: Any, predictions: list[dict[str, Any]], task: str) -> dict[str, float | None]:
    ground_truth = dataset.ground_truth()
    if not ground_truth or not predictions:
        return {"mAP": None, "IoU": None}
    if task in {"segment", "sam", "sam2"}:
        ious = [float(pred.get("mask_iou", 0.0)) for pred in predictions]
        return {"mAP": None, "IoU": sum(ious) / len(ious)}
    return {"mAP": _simple_map_proxy(ground_truth, predictions), "IoU": _mean_bbox_iou(ground_truth, predictions)}


def _mean_bbox_iou(gt: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> float:
    by_image = {}
    for ann in gt:
        by_image.setdefault(ann.get("image_id"), []).append(ann)
    scores = []
    for pred in predictions:
        candidates = by_image.get(pred.get("image_id"), [])
        if not candidates:
            continue
        scores.append(max(_bbox_iou(pred.get("bbox", []), ann.get("bbox", [])) for ann in candidates))
    return sum(scores) / len(scores) if scores else 0.0


def _simple_map_proxy(gt: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> float:
    matched = 0
    used = set()
    for pred in sorted(predictions, key=lambda item: item.get("score", 0), reverse=True):
        for idx, ann in enumerate(gt):
            if idx in used:
                continue
            if pred.get("category_id") == ann.get("category_id") and _bbox_iou(
                pred.get("bbox", []), ann.get("bbox", [])
            ) >= 0.5:
                matched += 1
                used.add(idx)
                break
    return matched / max(len(gt), 1)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

