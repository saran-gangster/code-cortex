from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.tensorrt.config import load_config
from deployment.tensorrt.postprocess import LEVELS, postprocess_fcos
from deployment.tensorrt.preprocess import ImageTransform, preprocess_image, preprocess_state

CONFIG = load_config(ROOT / "deployment" / "tensorrt" / "deployment.example.json")


def test_preprocessing_is_deterministic_and_profile_safe(tmp_path: Path) -> None:
    pixels = np.zeros((100, 180, 3), dtype=np.uint8)
    pixels[:, :, 0] = 255
    image_path = tmp_path / "frame.png"
    Image.fromarray(pixels).save(image_path)

    first, first_transform = preprocess_image(image_path, CONFIG)
    second, second_transform = preprocess_image(image_path, CONFIG)

    assert first.shape == (1, 3, 320, 576)
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    np.testing.assert_array_equal(first, second)
    assert first_transform == second_transform
    assert (first_transform.resized_height, first_transform.resized_width) == (320, 576)


def test_state_normalization_and_validation() -> None:
    state, mask = preprocess_state(CONFIG.state_mean, 0, CONFIG)

    np.testing.assert_allclose(state, 0.0, atol=1e-6)
    np.testing.assert_array_equal(mask, [[0.0]])
    with pytest.raises(ValueError, match="exactly 8"):
        preprocess_state([0.0] * 7, 1, CONFIG)
    with pytest.raises(ValueError, match="finite"):
        preprocess_state([float("nan")] + [0.0] * 7, 1, CONFIG)
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        preprocess_state([0.0] * 8, 0.5, CONFIG)


def test_raw_fcos_decode_and_class_aware_nms() -> None:
    outputs = {}
    for level in LEVELS:
        count = 1 if level in {"p3", "p4"} else 0
        logits = np.full((1, count, 9), -100.0, dtype=np.float32)
        if count:
            logits[0, 0, 2] = 10.0 if level == "p3" else 9.0
        outputs[f"cls_logits_{level}"] = logits
        outputs[f"bbox_regression_{level}"] = np.ones((1, count, 4), dtype=np.float32)
        outputs[f"bbox_ctrness_{level}"] = np.full((1, count, 1), 10.0, dtype=np.float32)
        outputs[f"anchors_{level}"] = np.tile(
            np.asarray([[[-4.0, -4.0, 4.0, 4.0]]], dtype=np.float32), (1, count, 1)
        )
    transform = ImageTransform(32, 32, 32, 32, 32, 32, 1.0, 1.0)

    detections = postprocess_fcos(outputs, transform, CONFIG)

    assert detections["boxes"].shape == (1, 4)
    np.testing.assert_allclose(detections["boxes"][0], [0.0, 0.0, 8.0, 8.0])
    np.testing.assert_array_equal(detections["labels"], [2])
    assert detections["scores"][0] > 0.99


def test_postprocess_rejects_wrong_runtime_shape() -> None:
    with pytest.raises(ValueError, match="missing"):
        postprocess_fcos({}, ImageTransform(1, 1, 1, 1, 32, 32, 1.0, 1.0), CONFIG)
