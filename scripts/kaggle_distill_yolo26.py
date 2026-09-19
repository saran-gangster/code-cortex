"""Distill the final AeroGuard YOLO26s detector into YOLO26n.

This script intentionally uses only the training and validation partitions. The
frozen final-test partition is not evaluated or used for model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES = ("Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = REPOSITORY_ROOT / "models" / "AeroGuard_YOLO26s_960.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill AeroGuard YOLO26s-960 into a smaller YOLO26n detector."
    )
    parser.add_argument("--data", type=Path, required=True, help="AU-AIR YOLO dataset YAML")
    parser.add_argument("--output", type=Path, required=True, help="Artifact output directory")
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--student", default="yolo26n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--hours",
        type=float,
        default=4.0,
        help="Wall-clock training limit; set to 0 to use only the epoch limit",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--distill-weight", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--name", default="yolo26n-960-distilled")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_dataset_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset YAML does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("dataset YAML must contain a mapping")
    for partition in ("train", "val"):
        if not payload.get(partition):
            raise ValueError(f"dataset YAML is missing the {partition!r} partition")
    if str(payload["train"]) == str(payload["val"]):
        raise ValueError("training and validation partitions must be different")

    raw_names = payload.get("names")
    if isinstance(raw_names, dict):
        names = tuple(value for _, value in sorted(raw_names.items(), key=lambda item: int(item[0])))
    elif isinstance(raw_names, list):
        names = tuple(raw_names)
    else:
        raise TypeError("dataset YAML must define class names as a list or indexed mapping")
    if names != CLASS_NAMES:
        raise ValueError(f"expected AU-AIR classes {CLASS_NAMES}, received {names}")
    return payload


def training_overrides(args: argparse.Namespace, teacher: Path, data: Path) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(args.output / "runs"),
        "name": args.name,
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 1e-3,
        "lrf": 1e-2,
        "weight_decay": 1e-3,
        "warmup_epochs": 1.0,
        "cos_lr": True,
        "amp": True,
        "deterministic": False,
        "seed": args.seed,
        "patience": 10,
        "close_mosaic": 2,
        "mosaic": 1.0,
        "mixup": 0.10,
        "multi_scale": 0.05,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.4,
        "degrees": 3.0,
        "translate": 0.10,
        "scale": 0.50,
        "fliplr": 0.5,
        "plots": True,
        "save": True,
        "verbose": True,
        "distill_model": str(teacher),
        "dis": args.distill_weight,
    }
    if args.hours > 0:
        overrides["time"] = args.hours
    return overrides


def require_distillation_support() -> None:
    from ultralytics.cfg import DEFAULT_CFG_DICT

    missing = {"distill_model", "dis"} - set(DEFAULT_CFG_DICT)
    if missing:
        fields = ", ".join(sorted(missing))
        raise RuntimeError(
            "the installed Ultralytics build does not support native YOLO26 distillation "
            f"(missing configuration fields: {fields})"
        )


def trained_checkpoint(trainer: Any) -> Path:
    for candidate in (getattr(trainer, "best", None), getattr(trainer, "last", None)):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("training completed without a best.pt or last.pt checkpoint")


def main(args: argparse.Namespace) -> None:
    teacher = args.teacher.resolve()
    data = args.data.resolve()
    args.output = args.output.resolve()
    if not teacher.is_file():
        raise FileNotFoundError(f"teacher checkpoint does not exist: {teacher}")
    student_source = Path(args.student)
    if student_source.is_file() and student_source.resolve() == teacher:
        raise ValueError("student and teacher checkpoints must be different")
    if args.imgsz <= 0 or args.imgsz % 32:
        raise ValueError("--imgsz must be a positive multiple of 32")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.distill_weight <= 0:
        raise ValueError("--distill-weight must be positive")
    validate_dataset_yaml(data)
    args.output.mkdir(parents=True, exist_ok=True)

    request = {
        "artifact_kind": "aeroguard_yolo26_feature_distillation",
        "teacher": str(teacher),
        "teacher_sha256": sha256(teacher),
        "student_initialization": args.student,
        "data_yaml": str(data),
        "image_size": args.imgsz,
        "distillation_weight": args.distill_weight,
        "epochs_ceiling": args.epochs,
        "hours_ceiling": args.hours if args.hours > 0 else None,
        "partitions_used": ["train", "validation"],
        "frozen_final_test_used": False,
        "started_unix": time.time(),
    }
    write_json(args.output / "distillation_request.json", request)

    from ultralytics import YOLO

    require_distillation_support()
    student = YOLO(args.student)
    started = time.perf_counter()
    student.train(**training_overrides(args, teacher, data))
    checkpoint = trained_checkpoint(student.trainer)

    final_checkpoint = args.output / f"AeroGuard_YOLO26n_{args.imgsz}_distilled.pt"
    shutil.copy2(checkpoint, final_checkpoint)

    selected = YOLO(str(final_checkpoint))
    validation = selected.val(
        data=str(data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=0.001,
        iou=0.6,
        plots=True,
        save_json=True,
        project=str(args.output / "evaluation"),
        name="distilled-validation",
    )
    validation_payload = {
        "results_dict": dict(validation.results_dict),
        "speed_ms": dict(validation.speed),
    }
    write_json(args.output / "validation_metrics.json", validation_payload)
    write_json(
        args.output / "distillation_summary.json",
        {
            **request,
            "student_checkpoint": str(final_checkpoint),
            "student_checkpoint_sha256": sha256(final_checkpoint),
            "source_training_checkpoint": str(checkpoint),
            "run_dir": str(student.trainer.save_dir),
            "elapsed_seconds": time.perf_counter() - started,
            "validation": validation_payload,
            "frozen_final_test_used": False,
            "finished_unix": time.time(),
        },
    )


if __name__ == "__main__":
    main(parse_args())
