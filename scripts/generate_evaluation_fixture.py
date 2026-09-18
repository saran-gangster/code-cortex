"""Generate non-benchmark evidence that the evaluation pipeline is executable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeroguard.evaluation import (
    build_evaluation_report,
    evaluate_calibration,
    evaluate_detections,
    match_visdrone_detections,
    write_evaluation_report,
)


def synthetic_records() -> list[dict]:
    return [
        {
            "image_id": "synthetic-a-1",
            "recording_root": "synthetic-a",
            "targets": [{"box_xyxy": [0, 0, 10, 10], "label": 1}],
            "predictions": [
                {"box_xyxy": [0, 0, 10, 10], "label": 1, "score": 0.9},
                {"box_xyxy": [0, 0, 10, 10], "label": 1, "score": 0.4},
            ],
        },
        {
            "image_id": "synthetic-b-1",
            "recording_root": "synthetic-b",
            "targets": [{"box_xyxy": [20, 20, 30, 30], "label": 2}],
            "predictions": [
                {"box_xyxy": [20, 20, 30, 30], "label": 2, "score": 0.8}
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("manifests/protocol.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation_pipeline_fixture.json"),
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))

    result = evaluate_detections(
        synthetic_records(),
        score_floor=0.05,
        max_detections_per_image=100,
        fixed_score_threshold=0.5,
    )
    calibration = evaluate_calibration([0.9, 0.8, 0.4], [True, True, False], bin_count=5)
    visdrone = match_visdrone_detections(
        [
            {"box_xyxy": [0, 0, 10, 10], "label": "Car", "score": 0.9},
            {"box_xyxy": [20, 20, 30, 30], "label": "Car", "score": 0.5},
        ],
        ["0,0,10,10,1,4,0,0", "20,20,10,10,0,0,0,0,"],
    )
    metrics = result.as_dict()
    metrics["calibration_fixture"] = calibration.to_dict()
    metrics["visdrone_ignore_policy_fixture"] = {
        "true_positives": visdrone.true_positives,
        "false_positives": visdrone.false_positives,
        "false_negatives": visdrone.false_negatives,
        "ignored_predictions": visdrone.ignored_predictions,
    }
    report = build_evaluation_report(
        artifact_kind="synthetic_evaluation_pipeline_fixture_not_benchmark",
        partition="synthetic_fixture",
        protocol_sha256=protocol["protocol_sha256"],
        final_test_unsealed=protocol["final_test_unsealed"],
        model_id="synthetic-evaluator-fixture",
        checkpoint_id="none-synthetic-fixture",
        config={
            "score_floor": 0.05,
            "fixed_score_threshold": 0.5,
            "max_detections_per_image": 100,
            "iou_thresholds": "0.50:0.05:0.95",
        },
        metrics=metrics,
        limitations=[
            "Synthetic boxes exercise software behavior only; they are not model results.",
            "No development, final-test, or VisDrone image was evaluated by this artifact.",
            "Calibration covers emitted detections only and does not measure missed objects.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_evaluation_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
