from datetime import datetime

import numpy as np
import pytest

from aeroguard.data import (
    STATE_FEATURE_NAMES,
    StateNormalizer,
    assert_protocol_is_leakage_safe,
    build_grouped_protocol,
    canonicalize_state,
    clean_boxes,
    parse_auair_record,
    reconstruct_timestamp,
)


def _record(name: str, *, ms: int = 0, boxes=None, altitude=2806.4, yaw=0.0):
    return {
        "image_name": name,
        "image_width:": 1920,
        "image_height": 1080,
        "time": {"year": 2019, "month": 8, "day": 29, "hour": 9, "min": 11, "sec": 11, "ms": ms},
        "altitude": altitude,
        "linear_x": 1.0,
        "linear_y": -2.0,
        "linear_z": 0.5,
        "angle_phi": 0.1,
        "angle_theta": -0.2,
        "angle_psi": yaw,
        "longtitude": 10.0,
        "latitude": 56.0,
        "bbox": boxes if boxes is not None else [{"top": 10, "left": 20, "height": 30, "width": 40, "class": 0}],
    }


def test_literal_aliases_box_conversion_and_label_offset():
    parsed = parse_auair_record(_record("frame_20190829091111_x_0000001.jpg"))
    assert parsed.image_width == 1920
    assert parsed.boxes == [(20.0, 10.0, 60.0, 40.0)]
    assert parsed.labels == [1]  # source Human=0, detector background=0
    assert parsed.image_name == "frame_20190829091111_x_0000001.jpg"
    assert parsed.recording_root == "20190829091111"


def test_box_cleaning_rejects_zero_size_and_keeps_empty_targets():
    result = clean_boxes(
        [{"top": 0, "left": 0, "height": 0, "width": 4, "class": 0}],
        image_width=100,
        image_height=100,
    )
    assert result.boxes == []
    assert result.labels == []
    assert result.rejected[0].reasons == ("non_positive_size",)
    parsed = parse_auair_record(
        _record("frame_20190829091111_x_0000002.jpg", boxes=[])
    )
    assert parsed.boxes == []
    assert parsed.labels == []


def test_timestamp_treats_ms_as_an_offset_not_a_fraction():
    timestamp = reconstruct_timestamp(
        {"year": 2020, "month": 1, "day": 2, "hour": 3, "min": 4, "sec": 5, "ms": 1500}
    )
    assert timestamp == datetime(2020, 1, 2, 3, 4, 6, 500_000)  # noqa: DTZ001


def test_state_units_and_circular_yaw_features():
    state = canonicalize_state(_record("frame_20190829091111_x_0000003.jpg", yaw=np.pi / 2))
    assert state["altitude_m"] == pytest.approx(2.8064)
    assert state["linear_y_mps"] == -2.0
    assert state["yaw_sin"] == pytest.approx(1.0)
    assert state["yaw_cos"] == pytest.approx(0.0, abs=1e-7)


def test_grouped_protocol_is_deterministic_and_keeps_x_xx_together():
    records = []
    for root in ("20190829091111", "20190901091111", "20190902091111", "20190903091111"):
        records.extend(
            [
                {"image_name": f"frame_{root}_x_0000001.jpg", "labels": [1]},
                {"image_name": f"frame_{root}_xx_0000002.jpg", "labels": [2]},
            ]
        )
    first = build_grouped_protocol(records, train_groups=2, development_groups=1, final_test_groups=1, seed=17)
    second = build_grouped_protocol(list(reversed(records)), train_groups=2, development_groups=1, final_test_groups=1, seed=17)
    assert first["partitions"] == second["partitions"]
    assert_protocol_is_leakage_safe(first)
    assert set(first["root_to_partition"]) == {record["image_name"].split("_")[1] for record in records}
    for record in records:
        assert first["record_to_partition"][record["image_name"]] == first["root_to_partition"][record["image_name"].split("_")[1]]


def test_state_normalizer_uses_training_ids_only_and_sanitizes_missing_values():
    def state(value):
        return {name: float(value) for name in STATE_FEATURE_NAMES}

    states = {"train-a": state(10), "train-b": state(20), "test": state(1000)}
    states["train-b"]["altitude_m"] = np.nan
    normalizer = StateNormalizer().fit(states, train_ids=["train-a", "train-b"])
    assert normalizer.mean[0] == pytest.approx(10.0)
    transformed = normalizer.transform([states["train-a"], states["test"]])
    assert np.isfinite(transformed).all()
    assert transformed[1, 0] > 100
    assert normalizer.availability_mask([states["train-a"], states["train-b"]]).tolist() == [True, False]
