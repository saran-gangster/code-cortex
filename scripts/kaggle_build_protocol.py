#!/usr/bin/env python3
"""Build the frozen AU-AIR manifest, grouped protocol, and train-only normalizer."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

FRAME_RE = re.compile(r"^frame_(\d{14})_(x|xx)_(\d+)\.jpg$")
STATE_NAMES = (
    "altitude_m",
    "linear_x_mps",
    "linear_y_mps",
    "linear_z_mps",
    "roll_rad",
    "pitch_rad",
    "yaw_sin",
    "yaw_cos",
)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reconstruct_time(raw: dict) -> datetime:
    # AU-AIR does not declare a timezone; preserving a naive source timestamp is intentional.
    base = datetime(  # noqa: DTZ001
        int(raw["year"]),
        int(raw["month"]),
        int(raw["day"]),
        int(raw["hour"]),
        int(raw["min"]),
        int(raw["sec"]),
    )
    return base + timedelta(milliseconds=float(raw["ms"]))


def canonical_state(raw: dict) -> list[float]:
    yaw = float(raw["angle_psi"])
    values = [
        float(raw["altitude"]) * 0.001,
        float(raw["linear_x"]),
        float(raw["linear_y"]),
        float(raw["linear_z"]),
        float(raw["angle_phi"]),
        float(raw["angle_theta"]),
        math.sin(yaw),
        math.cos(yaw),
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite canonical state")
    return values


def choose_protocol(records: list[dict], categories: list[str], seed: int) -> tuple[dict, dict]:
    by_root: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_root[record["recording_root"]].append(record)

    roots = sorted(by_root)
    if len(roots) != 8:
        raise ValueError(f"expected 8 recording roots, found {len(roots)}: {roots}")

    candidates = [
        (dev, tuple(sorted(test)))
        for dev in roots
        for test in itertools.combinations([root for root in roots if root != dev], 2)
    ]
    random.Random(seed).shuffle(candidates)

    selected = None
    selected_index = None
    for index, (dev, test) in enumerate(candidates):
        train = tuple(root for root in roots if root != dev and root not in test)
        train_labels = Counter(
            label
            for root in train
            for record in by_root[root]
            for label in record["model_labels"]
        )
        dev_labels = Counter(
            label for record in by_root[dev] for label in record["model_labels"]
        )
        final_frames = sum(len(by_root[root]) for root in test)
        # These rules use only pre-training support and frame counts, never model scores.
        if (
            len(train) == 5
            and all(train_labels[label] > 0 for label in range(1, len(categories) + 1))
            and len(by_root[dev]) >= 600
            and len(dev_labels) >= 4
            and final_frames >= 3000
        ):
            selected = {
                "train": list(train),
                "development": [dev],
                "final_test": list(test),
            }
            selected_index = index
            break
    if selected is None:
        raise RuntimeError("no split satisfied the declared support constraints")

    census: dict[str, dict] = {}
    for root in roots:
        counts = Counter(
            label for record in by_root[root] for label in record["model_labels"]
        )
        census[root] = {
            "frames": len(by_root[root]),
            "instances_by_model_label": {
                str(label): counts[label] for label in range(1, len(categories) + 1)
            },
        }

    protocol = {
        "schema_version": "1.0",
        "dataset": "AU-AIR",
        "seed": seed,
        "split_unit": "recording_root",
        "recording_root_pattern": FRAME_RE.pattern,
        "selection": selected,
        "selection_candidate_index_after_seeded_shuffle": selected_index,
        "selection_constraints": {
            "group_counts": {"train": 5, "development": 1, "final_test": 2},
            "all_classes_present_in_train": True,
            "minimum_development_frames": 600,
            "minimum_development_classes": 4,
            "minimum_final_test_frames": 3000,
            "model_scores_used": False,
        },
        "class_map": {
            "background_model_label": 0,
            "source_to_model_offset": 1,
            "source_categories_in_index_order": categories,
        },
        "root_census": census,
        "final_test_unsealed": False,
    }
    protocol_hash = sha256_bytes(canonical_json(protocol))
    protocol["protocol_sha256"] = protocol_hash
    return protocol, by_root


def fit_normalizer(records: list[dict], train_roots: set[str], protocol_hash: str) -> dict:
    train = [record["state"] for record in records if record["recording_root"] in train_roots]
    if not train:
        raise ValueError("cannot fit state normalizer without training records")
    means = [sum(row[i] for row in train) / len(train) for i in range(len(STATE_NAMES))]
    variances = [
        sum((row[i] - means[i]) ** 2 for row in train) / len(train)
        for i in range(len(STATE_NAMES))
    ]
    stds = [max(math.sqrt(value), 1e-8) for value in variances]
    payload = {
        "schema_version": "1.0",
        "feature_names": list(STATE_NAMES),
        "fit_partition": "train",
        "fit_recording_roots": sorted(train_roots),
        "sample_count": len(train),
        "mean": means,
        "std": stds,
        "protocol_sha256": protocol_hash,
    }
    payload["normalizer_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def build_records(annotation_path: Path) -> tuple[list[str], list[dict], list[dict]]:
    source = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = source["categories"]
    if categories != ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]:
        raise ValueError(f"unexpected source category order: {categories}")

    records = []
    rejected = []
    for raw in source["annotations"]:
        match = FRAME_RE.fullmatch(raw["image_name"])
        if not match:
            raise ValueError(f"invalid AU-AIR image identity: {raw['image_name']}")
        recording_root, stream, frame_index = match.groups()
        width = int(raw["image_width:"])
        height = int(raw["image_height"])
        boxes: list[list[float]] = []
        labels: list[int] = []
        for box_index, box in enumerate(raw["bbox"]):
            left = float(box["left"])
            top = float(box["top"])
            box_width = float(box["width"])
            box_height = float(box["height"])
            source_label = int(box["class"])
            reason = None
            if box_width <= 0 or box_height <= 0:
                reason = "non_positive_size"
            elif not 0 <= source_label < len(categories):
                reason = "class_out_of_range"
            else:
                x1 = max(0.0, min(left, float(width)))
                y1 = max(0.0, min(top, float(height)))
                x2 = max(0.0, min(left + box_width, float(width)))
                y2 = max(0.0, min(top + box_height, float(height)))
                if x2 <= x1 or y2 <= y1:
                    reason = "empty_after_bounds_clip"
            if reason:
                rejected.append(
                    {
                        "image_name": raw["image_name"],
                        "box_index": box_index,
                        "reason": reason,
                        "box": box,
                    }
                )
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(source_label + 1)

        timestamp = reconstruct_time(raw["time"])
        records.append(
            {
                "dataset_id": "AU-AIR",
                "frame_id": Path(raw["image_name"]).stem,
                "image_name": raw["image_name"],
                "image_relative_path": f"images/{raw['image_name']}",
                "recording_root": recording_root,
                "stream_id": stream,
                "frame_index": int(frame_index),
                "source_time_iso_no_timezone": timestamp.isoformat(),
                "original_size": {"width": width, "height": height},
                "boxes_xyxy": boxes,
                "model_labels": labels,
                "state": canonical_state(raw),
                "state_valid": True,
                "raw_state": {
                    "altitude_mm": raw["altitude"],
                    "linear_x_mps": raw["linear_x"],
                    "linear_y_mps": raw["linear_y"],
                    "linear_z_mps": raw["linear_z"],
                    "roll_rad": raw["angle_phi"],
                    "pitch_rad": raw["angle_theta"],
                    "yaw_rad": raw["angle_psi"],
                    "longitude_source_key_longtitude": raw["longtitude"],
                    "latitude": raw["latitude"],
                },
            }
        )
    records.sort(key=lambda item: (item["recording_root"], item["source_time_iso_no_timezone"], item["frame_index"]))
    return categories, records, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path("data/auair/04_AUAIR_multimodal_uav/annotations.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("manifests"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    categories, records, rejected = build_records(args.annotation)
    protocol, _ = choose_protocol(records, categories, args.seed)
    train_roots = set(protocol["selection"]["train"])
    normalizer = fit_normalizer(records, train_roots, protocol["protocol_sha256"])

    manifest_path = args.output / "auair.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output / "state_normalizer.json").write_text(
        json.dumps(normalizer, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output / "rejected_boxes.json").write_text(
        json.dumps(rejected, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    smoke_ids = [
        record["frame_id"]
        for record in records
        if record["recording_root"] in train_roots and record["model_labels"]
    ][:: max(1, sum(record["recording_root"] in train_roots for record in records) // 32)][:32]
    (args.output / "smoke_train_ids.json").write_text(
        json.dumps(smoke_ids, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "records": len(records),
        "valid_boxes": sum(len(record["boxes_xyxy"]) for record in records),
        "rejected_boxes": len(rejected),
        "recording_roots": sorted({record["recording_root"] for record in records}),
        "protocol_sha256": protocol["protocol_sha256"],
        "normalizer_sha256": normalizer["normalizer_sha256"],
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "smoke_frame_count": len(smoke_ids),
    }
    (args.output / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
