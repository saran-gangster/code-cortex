import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[1]


def test_computed_smoke_record_matches_versioned_schema():
    schema = json.loads((ROOT / "contracts" / "inference_record.schema.json").read_text())
    record = json.loads((ROOT / "reports" / "fcos_overfit_gate_inference.json").read_text())
    Draft202012Validator(schema).validate(record)


def test_frozen_protocol_has_no_group_overlap_and_is_still_sealed():
    protocol = json.loads((ROOT / "manifests" / "protocol.json").read_text())
    partitions = protocol["selection"]
    train, development, final_test = map(set, (partitions["train"], partitions["development"], partitions["final_test"]))
    assert len(train) == 5 and len(development) == 1 and len(final_test) == 2
    assert not (train & development or train & final_test or development & final_test)
    assert protocol["final_test_unsealed"] is False


def test_dual_gpu_engineering_arms_are_actually_matched():
    e2 = json.loads((ROOT / "reports" / "e2_rgb_masked_summary.json").read_text())
    e1 = json.loads((ROOT / "reports" / "e1_paired_film_summary.json").read_text())
    assert e2["shared_warmstart_sha256"] == e1["shared_warmstart_sha256"]
    assert e2["matched_frame_schedule_sha256"] == e1["matched_frame_schedule_sha256"]
    assert e2["frame_ids"] == e1["frame_ids"]
    assert {e2["physical_gpu_id"], e1["physical_gpu_id"]} == {"0", "1"}
    assert e2["state_mask"] == 0.0 and e2["film_projection_l1_after"] == 0.0
    assert e1["state_mask"] == 1.0 and e1["film_projection_l1_after"] > 0.0
    assert not e2["benchmark_claim"] and not e1["benchmark_claim"]


def test_random_development_controls_are_matched_and_train_only():
    protocol = json.loads((ROOT / "manifests" / "protocol.json").read_text())
    normalizer = json.loads((ROOT / "manifests" / "state_normalizer.json").read_text())
    e2 = json.loads((ROOT / "reports" / "development_random_e2_summary.json").read_text())
    e1 = json.loads((ROOT / "reports" / "development_random_e1_summary.json").read_text())

    assert e2["completed_steps"] == e1["completed_steps"] == 5000
    assert e2["training_record_count"] == e1["training_record_count"] == 18523
    assert e2["training_roots"] == e1["training_roots"] == protocol["selection"]["train"]
    assert e2["development_roots_used"] == e1["development_roots_used"] == []
    assert e2["final_test_roots_used"] == e1["final_test_roots_used"] == []
    assert e2["final_test_unsealed"] is e1["final_test_unsealed"] is False
    assert e2["shared_warmstart_sha256"] == e1["shared_warmstart_sha256"]
    assert e2["matched_frame_schedule_sha256"] == e1["matched_frame_schedule_sha256"]
    assert e2["normalizer_sha256"] == e1["normalizer_sha256"] == normalizer["normalizer_sha256"]
    assert e2["initial_weights_origin"] == e1["initial_weights_origin"] == "random_no_pretrained_weights"
    assert {e2["physical_gpu_id"], e1["physical_gpu_id"]} == {"0", "1"}
    assert e2["state_mask"] == 0.0 and e1["state_mask"] == 1.0
    assert not e2["benchmark_claim"] and not e1["benchmark_claim"]


def test_disclosed_imagenet_warmstart_is_nonbenchmark_and_hashed():
    summary = json.loads((ROOT / "reports" / "shared_imagenet_warmstart_summary.json").read_text())

    assert summary["weights_origin"] == "torchvision_resnet50_imagenet1k_v2_backbone"
    assert summary["torchvision_weights_enum"] == "ResNet50_Weights.IMAGENET1K_V2"
    assert len(summary["checkpoint_sha256"]) == 64
    assert summary["benchmark_claim"] is False


def test_imagenet_development_finetuning_is_matched_train_only_and_complete():
    protocol = json.loads((ROOT / "manifests" / "protocol.json").read_text())
    normalizer = json.loads((ROOT / "manifests" / "state_normalizer.json").read_text())
    warmstart = json.loads((ROOT / "reports" / "shared_imagenet_warmstart_summary.json").read_text())
    e2 = json.loads((ROOT / "reports" / "development_imagenet_e2_summary.json").read_text())
    e1 = json.loads((ROOT / "reports" / "development_imagenet_e1_summary.json").read_text())

    assert e2["completed_steps"] == e1["completed_steps"] == 5000
    assert e2["training_record_count"] == e1["training_record_count"] == 18523
    assert e2["training_roots"] == e1["training_roots"] == protocol["selection"]["train"]
    assert e2["development_roots_used"] == e1["development_roots_used"] == []
    assert e2["final_test_roots_used"] == e1["final_test_roots_used"] == []
    assert e2["final_test_unsealed"] is e1["final_test_unsealed"] is False
    assert e2["shared_warmstart_sha256"] == e1["shared_warmstart_sha256"] == warmstart["checkpoint_sha256"]
    assert e2["matched_frame_schedule_sha256"] == e1["matched_frame_schedule_sha256"]
    assert e2["normalizer_sha256"] == e1["normalizer_sha256"] == normalizer["normalizer_sha256"]
    assert e2["initial_weights_origin"] == e1["initial_weights_origin"] == warmstart["weights_origin"]
    assert {e2["physical_gpu_id"], e1["physical_gpu_id"]} == {"0", "1"}
    assert e2["state_mask"] == 0.0 and e1["state_mask"] == 1.0
    assert not e2["benchmark_claim"] and not e1["benchmark_claim"]
