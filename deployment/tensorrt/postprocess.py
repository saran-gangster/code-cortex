"""NumPy implementation of TorchVision FCOS decode and class-aware NMS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .config import DeploymentConfig
from .preprocess import ImageTransform

LEVELS = ("p3", "p4", "p5", "p6", "p7")


def output_names() -> tuple[str, ...]:
    return tuple(
        name
        for level in LEVELS
        for name in (
            f"cls_logits_{level}",
            f"bbox_regression_{level}",
            f"bbox_ctrness_{level}",
            f"anchors_{level}",
        )
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _decode(regression: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    centers_x = 0.5 * (anchors[:, 0] + anchors[:, 2])
    centers_y = 0.5 * (anchors[:, 1] + anchors[:, 3])
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    scaled = regression * np.stack((widths, heights, widths, heights), axis=1)
    return np.stack(
        (
            centers_x - scaled[:, 0],
            centers_y - scaled[:, 1],
            centers_x + scaled[:, 2],
            centers_y + scaled[:, 3],
        ),
        axis=1,
    ).astype(np.float32)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(right_bottom - left_top, 0.0), axis=1)
    area_a = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / np.maximum(area_a + area_b - intersection, np.finfo(np.float32).eps)


def _batched_nms(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, threshold: float, limit: int
) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    retained: list[int] = []
    while order.size and len(retained) < limit:
        current = int(order[0])
        retained.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        overlaps = _iou(boxes[current], boxes[remaining])
        suppress = (labels[remaining] == labels[current]) & (overlaps > threshold)
        order = remaining[~suppress]
    return np.asarray(retained, dtype=np.int64)


def _validated_output(value: np.ndarray, name: str, final: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != final:
        raise ValueError(f"{name} must have shape [1, anchors, {final}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array[0].astype(np.float32, copy=False)


def postprocess_fcos(
    outputs: Mapping[str, np.ndarray], transform: ImageTransform, config: DeploymentConfig
) -> dict[str, np.ndarray]:
    missing = sorted(set(output_names()) - set(outputs))
    if missing:
        raise ValueError(f"runtime outputs are missing: {', '.join(missing)}")
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for level in LEVELS:
        logits = _validated_output(outputs[f"cls_logits_{level}"], f"cls_logits_{level}", 9)
        regression = _validated_output(
            outputs[f"bbox_regression_{level}"], f"bbox_regression_{level}", 4
        )
        centerness = _validated_output(
            outputs[f"bbox_ctrness_{level}"], f"bbox_ctrness_{level}", 1
        )
        anchors = _validated_output(outputs[f"anchors_{level}"], f"anchors_{level}", 4)
        if not (len(logits) == len(regression) == len(centerness) == len(anchors)):
            raise ValueError(f"inconsistent anchor counts at {level}")
        scores = np.sqrt(_sigmoid(logits) * _sigmoid(centerness)).reshape(-1)
        candidates = np.flatnonzero(scores > config.score_threshold)
        if candidates.size == 0:
            continue
        candidate_scores = scores[candidates]
        rank = np.argsort(-candidate_scores, kind="stable")[: config.topk_candidates]
        flat_indices = candidates[rank]
        candidate_scores = candidate_scores[rank]
        anchor_indices = flat_indices // 9
        labels = flat_indices % 9
        boxes = _decode(regression[anchor_indices], anchors[anchor_indices])
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0.0, transform.resized_width)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0.0, transform.resized_height)
        all_boxes.append(boxes)
        all_scores.append(candidate_scores.astype(np.float32))
        all_labels.append(labels.astype(np.int64))

    if not all_boxes:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
        }
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    keep = _batched_nms(
        boxes, scores, labels, config.nms_threshold, config.detections_per_image
    )
    boxes = boxes[keep]
    boxes[:, (0, 2)] /= transform.scale_x
    boxes[:, (1, 3)] /= transform.scale_y
    boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0.0, transform.original_width)
    boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0.0, transform.original_height)
    return {"boxes": boxes, "scores": scores[keep], "labels": labels[keep]}


def serialize_detections(detections: Mapping[str, Sequence[object] | np.ndarray]) -> list[dict[str, object]]:
    boxes = np.asarray(detections["boxes"])
    scores = np.asarray(detections["scores"])
    labels = np.asarray(detections["labels"])
    return [
        {"box_xyxy": [float(item) for item in box], "score": float(score), "label": int(label)}
        for box, score, label in zip(boxes, scores, labels, strict=True)
    ]


__all__ = ["LEVELS", "output_names", "postprocess_fcos", "serialize_detections"]
