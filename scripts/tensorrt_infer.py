"""Run batch-one AeroGuard edge inference from a TensorRT engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.tensorrt.config import load_config
from deployment.tensorrt.metadata import (
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)
from deployment.tensorrt.postprocess import postprocess_fcos, serialize_detections
from deployment.tensorrt.preprocess import preprocess_image, preprocess_state
from deployment.tensorrt.runtime import TensorRTRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AeroGuard TensorRT edge inference.")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--state", type=float, nargs=8, required=True, metavar="VALUE")
    parser.add_argument("--state-valid", type=int, choices=(0, 1), required=True)
    parser.add_argument("--metadata", type=Path, help="Engine metadata; defaults to ENGINE.json")
    parser.add_argument("--output", type=Path, help="Write JSON result instead of stdout only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    engine = args.engine.expanduser().resolve()
    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata
        else engine.with_suffix(engine.suffix + ".json")
    )
    if not metadata_path.is_file():
        raise RuntimeError(f"engine provenance metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("engine_sha256") != sha256_file(engine):
        raise RuntimeError("engine SHA-256 does not match its provenance metadata")
    if metadata.get("config_sha256") != canonical_sha256(config.raw):
        raise RuntimeError("deployment config does not match the engine build metadata")
    images, transform = preprocess_image(args.image, config)
    state, state_mask = preprocess_state(args.state, args.state_valid, config)
    raw = TensorRTRunner(engine).infer(
        {"images": images, "state": state, "state_mask": state_mask}
    )
    detections = postprocess_fcos(raw, transform, config)
    result = {
        "artifact_kind": "aeroguard_tensorrt_inference",
        "engine_sha256": metadata["engine_sha256"],
        "image": str(args.image.expanduser().resolve()),
        "state_valid": bool(args.state_valid),
        "preprocessed_shape": list(images.shape),
        "detections": serialize_detections(detections),
    }
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
