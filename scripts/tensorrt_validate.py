"""Compare raw ONNX Runtime or TensorRT outputs with the PyTorch export wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.tensorrt.config import load_config
from deployment.tensorrt.model import (
    create_raw_export_model,
    load_flight_aware_model,
    verify_checkpoint_provenance,
)
from deployment.tensorrt.postprocess import output_names
from deployment.tensorrt.preprocess import preprocess_image, preprocess_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exported AeroGuard runtime outputs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--state", type=float, nargs=8, required=True, metavar="VALUE")
    parser.add_argument("--state-valid", type=int, choices=(0, 1), required=True)
    parser.add_argument("--backend", choices=("onnxruntime", "tensorrt"), required=True)
    parser.add_argument("--artifact", type=Path, required=True, help="ONNX model or TensorRT engine")
    parser.add_argument("--run-summary", type=Path)
    parser.add_argument("--allow-unverified-provenance", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def _runtime_outputs(args, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if args.backend == "onnxruntime":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNX validation") from exc
        session = ort.InferenceSession(str(args.artifact), providers=["CPUExecutionProvider"])
        values = session.run(list(output_names()), feeds)
        return dict(zip(output_names(), values, strict=True))
    from deployment.tensorrt.runtime import TensorRTRunner

    return TensorRTRunner(args.artifact).infer(feeds)


def main() -> None:
    args = parse_args()
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("tolerances must be non-negative")
    config = load_config(args.config)
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha, _ = verify_checkpoint_provenance(
        checkpoint,
        args.run_summary,
        allow_unverified=args.allow_unverified_provenance,
    )
    images, _ = preprocess_image(args.image, config)
    state, state_mask = preprocess_state(args.state, args.state_valid, config)
    feeds = {"images": images, "state": state, "state_mask": state_mask}
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the validation reference") from exc
    wrapper = create_raw_export_model(load_flight_aware_model(checkpoint, config))
    with torch.inference_mode():
        values = wrapper(*(torch.from_numpy(feeds[name]) for name in ("images", "state", "state_mask")))
    reference = {
        name: value.detach().cpu().numpy()
        for name, value in zip(output_names(), values, strict=True)
    }
    candidate = _runtime_outputs(args, feeds)
    comparisons = {}
    passed = True
    for name in output_names():
        if name not in candidate:
            comparisons[name] = {"passed": False, "error": "missing output"}
            passed = False
            continue
        expected = reference[name].astype(np.float32)
        actual = np.asarray(candidate[name], dtype=np.float32)
        if expected.shape != actual.shape:
            comparisons[name] = {
                "passed": False,
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
            }
            passed = False
            continue
        absolute = np.abs(expected - actual)
        denominator = np.maximum(np.abs(expected), np.finfo(np.float32).eps)
        item_passed = bool(np.allclose(expected, actual, atol=args.atol, rtol=args.rtol))
        comparisons[name] = {
            "passed": item_passed,
            "max_absolute_error": float(absolute.max(initial=0.0)),
            "max_relative_error": float((absolute / denominator).max(initial=0.0)),
        }
        passed &= item_passed
    report = {
        "artifact_kind": "aeroguard_export_validation",
        "backend": args.backend,
        "checkpoint_sha256": checkpoint_sha,
        "atol": args.atol,
        "rtol": args.rtol,
        "passed": passed,
        "outputs": comparisons,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
