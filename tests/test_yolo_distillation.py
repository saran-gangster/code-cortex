from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from scripts.kaggle_distill_yolo26 import (
    CLASS_NAMES,
    training_overrides,
    validate_dataset_yaml,
)


def _dataset_yaml(path: Path, *, names: list[str] | None = None) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(path.parent / "dataset"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names or list(CLASS_NAMES),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_distillation_uses_native_yolo26_teacher_without_opening_test(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.pt"
    data = _dataset_yaml(tmp_path / "data.yaml")
    args = argparse.Namespace(
        epochs=50,
        imgsz=960,
        batch=8,
        device="0",
        workers=6,
        output=tmp_path / "output",
        name="yolo26n-960-distilled",
        seed=1337,
        hours=4.0,
        distill_weight=6.0,
    )

    overrides = training_overrides(args, teacher, data)

    assert overrides["distill_model"] == str(teacher)
    assert overrides["dis"] == 6.0
    assert overrides["data"] == str(data)
    assert "split" not in overrides
    assert "test" not in overrides


def test_dataset_contract_requires_the_eight_auair_classes(tmp_path: Path) -> None:
    data = _dataset_yaml(tmp_path / "data.yaml", names=["Car"])

    with pytest.raises(ValueError, match="expected AU-AIR classes"):
        validate_dataset_yaml(data)


def test_dataset_contract_rejects_train_validation_leakage(tmp_path: Path) -> None:
    data = _dataset_yaml(tmp_path / "data.yaml")
    payload = yaml.safe_load(data.read_text(encoding="utf-8"))
    payload["val"] = payload["train"]
    data.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be different"):
        validate_dataset_yaml(data)
