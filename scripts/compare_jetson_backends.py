"""Compare per-frame detections from two Jetson benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def box_iou(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))

    ious: list[float] = []
    confidence_deltas: list[float] = []
    matched = extra = missing = 0
    frames = []
    for reference_frame, candidate_frame in zip(
        reference["detections"], candidate["detections"], strict=True
    ):
        used: set[int] = set()
        frame_ious = []
        for reference_detection in reference_frame["detections"]:
            candidates = [
                (
                    box_iou(
                        reference_detection["box_xyxy"],
                        candidate_detection["box_xyxy"],
                    ),
                    index,
                    candidate_detection,
                )
                for index, candidate_detection in enumerate(candidate_frame["detections"])
                if index not in used
                and candidate_detection["class_id"] == reference_detection["class_id"]
            ]
            if not candidates:
                missing += 1
                continue
            iou, index, candidate_detection = max(candidates)
            used.add(index)
            matched += 1
            ious.append(iou)
            frame_ious.append(iou)
            confidence_deltas.append(
                abs(
                    reference_detection["confidence"]
                    - candidate_detection["confidence"]
                )
            )
        frame_extra = len(candidate_frame["detections"]) - len(used)
        extra += frame_extra
        frames.append(
            {
                "image": reference_frame["image"],
                "reference_count": reference_frame["count"],
                "candidate_count": candidate_frame["count"],
                "matched_ious": frame_ious,
                "extra_candidate_detections": frame_extra,
            }
        )

    payload = {
        "matched_detections": matched,
        "extra_candidate_detections": extra,
        "missing_candidate_detections": missing,
        "mean_iou": sum(ious) / len(ious),
        "minimum_iou": min(ious),
        "mean_absolute_confidence_delta": sum(confidence_deltas)
        / len(confidence_deltas),
        "maximum_absolute_confidence_delta": max(confidence_deltas),
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
