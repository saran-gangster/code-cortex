from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.tensorrt.config import ConfigError, load_config, validate_config

EXAMPLE = ROOT / "deployment" / "tensorrt" / "deployment.example.json"


def test_example_deployment_config_is_valid_and_dynamic() -> None:
    config = load_config(EXAMPLE)

    assert config.state_dim == 8
    assert config.precision == "fp16"
    assert config.image_profile.dynamic is True
    assert config.image_profile.maximum == (1, 3, 576, 576)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data["build"].update(precision="int8"), "INT8"),
        (
            lambda data: data["build"]["profiles"]["images"].update(
                min=[1, 3, 321, 320]
            ),
            "multiples of 32",
        ),
        (lambda data: data["model"].update(state_dim=7), "state_dim=8"),
        (
            lambda data: data["preprocessing"]["state_feature_names"].reverse(),
            "trained feature order",
        ),
        (lambda data: data["build"]["profiles"].update(state=[2, 8]), "fixed at"),
    ],
)
def test_invalid_contracts_fail_clearly(mutate, message: str) -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    mutate(data)

    with pytest.raises(ConfigError, match=message):
        validate_config(data)


def test_fixed_shape_contract_is_recognized() -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    fixed = [1, 3, 576, 576]
    data["build"]["profiles"]["images"] = {
        "min": copy.copy(fixed),
        "opt": copy.copy(fixed),
        "max": copy.copy(fixed),
    }

    assert validate_config(data).image_profile.dynamic is False
