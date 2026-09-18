import copy
import json
from pathlib import Path

import pytest

from aeroguard.data.provenance import validate_protocol_bundle

ROOT = Path(__file__).resolve().parents[1]


def bundle():
    protocol = json.loads((ROOT / "manifests" / "protocol.json").read_text())
    normalizer = json.loads((ROOT / "manifests" / "state_normalizer.json").read_text())
    return protocol, normalizer


def test_checked_in_protocol_and_normalizer_are_cryptographically_consistent():
    protocol, normalizer = bundle()
    partitions = validate_protocol_bundle(protocol, normalizer)

    assert len(partitions.train) == 5
    assert len(partitions.development) == 1
    assert len(partitions.final_test) == 2


def test_overlap_and_wrong_fit_roots_are_rejected():
    protocol, normalizer = bundle()
    overlapping = copy.deepcopy(protocol)
    overlapping["selection"]["development"] = [overlapping["selection"]["train"][0]]
    with pytest.raises(ValueError, match="overlap"):
        validate_protocol_bundle(overlapping, normalizer)

    wrong_roots = copy.deepcopy(normalizer)
    wrong_roots["fit_recording_roots"] = protocol["selection"]["development"]
    with pytest.raises(ValueError, match="fit roots"):
        validate_protocol_bundle(protocol, wrong_roots)


def test_tampered_normalizer_values_are_rejected_even_if_labels_are_unchanged():
    protocol, normalizer = bundle()
    tampered = copy.deepcopy(normalizer)
    tampered["mean"][0] += 1.0

    with pytest.raises(ValueError, match="content hash"):
        validate_protocol_bundle(protocol, tampered)
