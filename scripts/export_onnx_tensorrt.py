"""Export the trusted AeroGuard Lightning checkpoint as raw pre-NMS ONNX."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.tensorrt.config import load_config
from deployment.tensorrt.metadata import (
    canonical_sha256,
    require_nonempty_artifact,
    sha256_file,
    write_json_atomic,
)
from deployment.tensorrt.model import (
    create_raw_export_model,
    load_flight_aware_model,
    verify_checkpoint_provenance,
)
from deployment.tensorrt.postprocess import output_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export AeroGuard FCOS + gated FiLM to raw pre-NMS ONNX."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path)
    parser.add_argument(
        "--allow-unverified-provenance",
        action="store_true",
        help="Permit a trusted local .ckpt without run_summary.json (recorded in metadata).",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.opset < 17:
        raise ValueError("opset must be at least 17 for this export contract")
    config = load_config(args.config)
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha, summary = verify_checkpoint_provenance(
        checkpoint,
        args.run_summary,
        allow_unverified=args.allow_unverified_provenance,
    )
    try:
        import onnx
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires torch, torchvision, and onnx; install the ML extra and onnx"
        ) from exc

    model = create_raw_export_model(load_flight_aware_model(checkpoint, config))
    sample_shape = config.image_profile.optimum
    images = torch.zeros(sample_shape, dtype=torch.float32)
    state = torch.zeros((1, config.state_dim), dtype=torch.float32)
    state_mask = torch.ones((1, 1), dtype=torch.float32)
    dynamic_axes = None
    if config.image_profile.dynamic:
        dynamic_axes = {
            "images": {2: "image_height", 3: "image_width"},
            **{name: {1: f"{name}_anchors"} for name in output_names()},
        }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (images, state, state_mask),
            temporary,
            input_names=["images", "state", "state_mask"],
            output_names=list(output_names()),
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    exported = onnx.load(str(temporary), load_external_data=True)
    onnx.checker.check_model(exported, full_check=True)
    temporary.replace(output)
    require_nonempty_artifact(output, "ONNX model")
    metadata = {
        "artifact_kind": "aeroguard_raw_pre_nms_onnx",
        "schema_version": 1,
        "onnx_path": str(output),
        "onnx_sha256": sha256_file(output),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_provenance_verified": summary is not None,
        "experiment_arm": summary.get("experiment_arm") if summary else None,
        "config_path": str(config.source),
        "config_sha256": canonical_sha256(config.raw),
        "opset": args.opset,
        "shape_mode": "dynamic_hw_batch_1" if config.image_profile.dynamic else "fixed_batch_1",
        "inputs": {"images": list(sample_shape), "state": [1, 8], "state_mask": [1, 1]},
        "outputs": list(output_names()),
        "postprocessing_embedded": False,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "onnx_version": onnx.__version__,
        "python_version": platform.python_version(),
    }
    write_json_atomic(output.with_suffix(output.suffix + ".json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
