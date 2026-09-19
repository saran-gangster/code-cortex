from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.generate_final_benchmark_summary import extract, plot, write_csv


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_final_benchmark_summary_uses_validation_for_selection_artifacts(tmp_path: Path) -> None:
    candidate = tmp_path / "yolo26s-960"
    candidate.mkdir()
    counts = {
        "precision": 0.5,
        "recall": 0.4,
        "f1": 0.444,
        "detection_accuracy": 0.286,
        "true_positives": 40,
        "false_positives": 40,
        "false_negatives": 60,
    }
    _write_json(
        candidate / "final_summary.json",
        {
            "checkpoint": "best.pt",
            "validation": {
                "results_dict": {"metrics/mAP50(B)": 0.31, "metrics/mAP50-95(B)": 0.14},
                "operating_point_counts": counts,
            },
            "final_test": {
                "results_dict": {"metrics/mAP50(B)": 0.28, "metrics/mAP50-95(B)": 0.12},
                "operating_point_counts": counts,
            },
            "test_loss": {
                "partition": "test",
                "images": 1580,
                "test/box_loss": 1.0,
                "test/cls_loss": 0.8,
                "test/l1_loss": 0.01,
                "test_loss_total": 1.81,
            },
        },
    )
    _write_json(candidate / "run_request.json", {"model": "yolo26s.pt", "imgsz": 960})
    _write_json(
        candidate / "epoch_history.json",
        {
            "epochs": [1, 2],
            "training_total_loss": [4.5, 3.9],
            "validation_total_loss": [4.4, 4.0],
        },
    )

    result = extract(candidate)

    assert result["model"] == "yolo26s.pt"
    assert result["validation"]["ap50_95"] == 0.14
    assert result["final_test"]["loss"]["partition"] == "test"
    assert result["loss_curve"]["final_test"] == 1.81

    csv_path = tmp_path / "comparison.csv"
    write_csv(csv_path, [result], result["candidate"])
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows[0]["selected"] == "True"
    assert rows[0]["validation_ap50_95"] == "0.14"

    graph_path = tmp_path / "comparison.png"
    plot(graph_path, [result])
    assert graph_path.stat().st_size > 1_000
