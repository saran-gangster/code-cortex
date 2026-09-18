import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "reports" / "review2"
EVALUATION_ROOT = ROOT / "reports" / "evaluations"
RUN_NAMES = ("imagenet-e2-full-pass", "imagenet-e1-full-pass")
ADAPTER_NAMES = (
    "imagenet-e3-exact-frozen-visual-film",
    "imagenet-e4-exact-frozen-visual-film-dropout50",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_review2_full_pass_runs_are_complete_matched_and_hashed():
    summaries = [read_json(RUN_ROOT / name / "run_summary.json") for name in RUN_NAMES]

    for name, summary in zip(RUN_NAMES, summaries, strict=True):
        history_path = RUN_ROOT / name / "loss_history.jsonl"
        history = read_jsonl(history_path)
        assert summary["completed_steps"] == summary["requested_steps"] == 18_523
        assert summary["unique_training_frames_presented"] == 18_523
        assert summary["training_frame_coverage_fraction"] == 1.0
        assert summary["loss_history_rows"] == len(history) == 18_523
        assert [row["step"] for row in history] == list(range(1, 18_524))
        assert sha256_file(history_path) == summary["loss_history_sha256"]
        assert summary["final_test_unsealed"] is False
        assert summary["development_roots_used"] == []
        assert summary["final_test_roots_used"] == []

    for field in (
        "matched_frame_schedule_sha256",
        "shared_warmstart_sha256",
        "protocol_sha256",
        "normalizer_sha256",
    ):
        assert summaries[0][field] == summaries[1][field]
    assert {summary["physical_gpu_id"] for summary in summaries} == {"0", "1"}
    assert summaries[0]["state_mask"] == 0.0
    assert summaries[1]["state_mask"] == 1.0


def test_review2_evaluations_are_development_only_and_traceable():
    summaries = {
        name: read_json(RUN_ROOT / name / "run_summary.json") for name in RUN_NAMES
    }
    reports = {
        name: read_json(EVALUATION_ROOT / f"review2-{name}.json")
        for name in (
            *RUN_NAMES,
            "imagenet-e1-masked-fallback",
            "imagenet-e1-shuffled-state",
        )
    }

    for name, report in reports.items():
        expected_run = RUN_NAMES[0] if name == RUN_NAMES[0] else RUN_NAMES[1]
        assert report["partition"] == "development"
        assert report["benchmark_claim"] is False
        assert report["final_test_unsealed"] is False
        assert report["checkpoint_id"] == summaries[expected_run]["checkpoint_sha256"]
        assert report["metrics"]["evaluated_frame_count"] == 5_734
        assert report["metrics"]["evaluated_object_count"] > 0
        sweep = report["metrics"]["operating_point_sweep"]
        assert sweep["selection_partition"] == "development"
        assert sweep["accuracy_definition"].startswith("TP / (TP + FP + FN)")
        assert len(sweep["points"]) == 19
        assert sweep["best_f1_point"] in sweep["points"]

    assert reports[RUN_NAMES[0]]["config"]["state_mode"] == "masked"
    assert reports[RUN_NAMES[1]]["config"]["state_mode"] == "paired"
    assert reports["imagenet-e1-masked-fallback"]["config"]["state_mode"] == "masked"
    assert reports["imagenet-e1-shuffled-state"]["config"]["state_alignment"] == (
        "deliberately_1000_frame_shifted"
    )


def test_corrected_model_labels_match_full_pass_metrics():
    e2 = read_json(EVALUATION_ROOT / "review2-imagenet-e2-full-pass.json")
    e1 = read_json(EVALUATION_ROOT / "review2-imagenet-e1-full-pass.json")

    assert e2["model_id"] == "imagenet-full-pass-e2-rgb-masked"
    assert e1["model_id"] == "imagenet-full-pass-e1-paired-film"
    assert e2["metrics"]["pooled"]["ap50"] > e1["metrics"]["pooled"]["ap50"]
    assert (
        e2["metrics"]["operating_point_sweep"]["best_f1_point"]["f1"]
        > e1["metrics"]["operating_point_sweep"]["best_f1_point"]["f1"]
    )


def test_review2_frozen_visual_adapters_are_complete_and_traceable():
    e2_summary = read_json(RUN_ROOT / RUN_NAMES[0] / "run_summary.json")
    warmstart = read_json(RUN_ROOT / "e2-frozen-visual-warmstart-summary.json")
    summaries = {
        name: read_json(RUN_ROOT / name / "run_summary.json")
        for name in ADAPTER_NAMES
    }

    assert warmstart["source_checkpoint_sha256"] == e2_summary["checkpoint_sha256"]
    assert warmstart["visual_detector_will_be_frozen"] is True
    for name, summary in summaries.items():
        history_path = RUN_ROOT / name / "loss_history.jsonl"
        history = read_jsonl(history_path)
        assert summary["completed_steps"] == summary["requested_steps"] == 18_523
        assert summary["unique_training_frames_presented"] == 18_523
        assert summary["training_frame_coverage_fraction"] == 1.0
        assert summary["loss_history_rows"] == len(history) == 18_523
        assert [row["step"] for row in history] == list(range(1, 18_524))
        assert sha256_file(history_path) == summary["loss_history_sha256"]
        assert summary["shared_warmstart_sha256"] == warmstart["checkpoint_sha256"]
        assert summary["visual_detector_frozen"] is True
        assert summary["visual_running_statistics_frozen"] is True
        assert summary["final_test_unsealed"] is False
        assert summary["development_roots_used"] == []
        assert summary["final_test_roots_used"] == []

    assert summaries[ADAPTER_NAMES[0]]["matched_frame_schedule_sha256"] == (
        summaries[ADAPTER_NAMES[1]]["matched_frame_schedule_sha256"]
    )
    assert summaries[ADAPTER_NAMES[0]]["effective_state_mask_mean"] == 1.0
    assert 0.49 <= summaries[ADAPTER_NAMES[1]]["effective_state_mask_mean"] <= 0.51
    assert {summary["physical_gpu_id"] for summary in summaries.values()} == {"0", "1"}


def test_review2_adapter_integrity_and_evaluations_are_sealed():
    summaries = {
        name: read_json(RUN_ROOT / name / "run_summary.json")
        for name in ADAPTER_NAMES
    }
    integrity = read_json(RUN_ROOT / "adapter_integrity_summary.json")

    assert integrity["all_visual_tensors_exactly_equal"] is True
    assert integrity["final_test_unsealed"] is False
    for name in ADAPTER_NAMES:
        run = integrity["runs"][name]
        report = read_json(EVALUATION_ROOT / f"review2-{name}.json")
        assert run["visual_tensors_exactly_equal"] is True
        assert run["visual_state_sha256"] == integrity["parent_visual_state_sha256"]
        assert run["changed_film_tensor_count"] > 0
        assert report["partition"] == "development"
        assert report["benchmark_claim"] is False
        assert report["final_test_unsealed"] is False
        assert report["checkpoint_id"] == summaries[name]["checkpoint_sha256"]
        assert report["metrics"]["evaluated_frame_count"] == 5_734
