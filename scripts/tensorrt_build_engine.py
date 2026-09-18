"""Build a real TensorRT engine with trtexec or the TensorRT Python API."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AeroGuard TensorRT engine from ONNX.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "trtexec", "python"), default="auto")
    parser.add_argument("--trtexec", help="Explicit trtexec executable path")
    parser.add_argument("--dry-run", action="store_true", help="Print the trtexec command only")
    return parser.parse_args()


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(item) for item in shape)


def _trtexec_command(executable: str, onnx: Path, output: Path, config) -> list[str]:
    profile = config.image_profile
    fixed_inputs = ",state:1x8,state_mask:1x1"
    command = [
        executable,
        f"--onnx={onnx}",
        f"--saveEngine={output}",
        f"--minShapes=images:{_shape_text(profile.minimum)}{fixed_inputs}",
        f"--optShapes=images:{_shape_text(profile.optimum)}{fixed_inputs}",
        f"--maxShapes=images:{_shape_text(profile.maximum)}{fixed_inputs}",
        f"--memPoolSize=workspace:{config.workspace_mib}MiB",
        "--skipInference",
    ]
    if config.precision == "fp16":
        command.append("--fp16")
    return command


def _build_python(onnx: Path, output: Path, config) -> str:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python bindings are unavailable") from exc
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parser failed:\n{errors}")
    build_config = builder.create_builder_config()
    if hasattr(build_config, "set_memory_pool_limit"):
        build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, config.workspace_mib << 20)
    else:
        build_config.max_workspace_size = config.workspace_mib << 20
    if config.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("FP16 requested but this TensorRT platform does not report fast FP16")
        build_config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        config.image_profile.minimum,
        config.image_profile.optimum,
        config.image_profile.maximum,
    )
    profile.set_shape("state", (1, 8), (1, 8), (1, 8))
    profile.set_shape("state_mask", (1, 1), (1, 1), (1, 1))
    build_config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned no serialized engine")
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(bytes(serialized))
    os.replace(temporary, output)
    return trt.__version__


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    onnx = require_nonempty_artifact(args.onnx, "ONNX model")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    executable = args.trtexec or shutil.which("trtexec")
    backend = args.backend
    if backend == "auto":
        backend = "trtexec" if executable else "python"
    command = None
    version = None
    if backend == "trtexec":
        if not executable:
            raise RuntimeError("trtexec was not found; pass --trtexec or use --backend python")
        command = _trtexec_command(executable, onnx, output, config)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            return
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise RuntimeError(f"trtexec failed with exit code {completed.returncode}")
        version_result = subprocess.run(
            [executable, "--version"], check=False, capture_output=True, text=True
        )
        version = (version_result.stdout or version_result.stderr).strip()
    else:
        if args.dry_run:
            raise ValueError("--dry-run is supported only with the trtexec backend")
        version = _build_python(onnx, output, config)
    require_nonempty_artifact(output, "TensorRT engine")
    metadata = {
        "artifact_kind": "aeroguard_tensorrt_engine",
        "schema_version": 1,
        "engine_path": str(output),
        "engine_sha256": sha256_file(output),
        "onnx_path": str(onnx),
        "onnx_sha256": sha256_file(onnx),
        "config_path": str(config.source),
        "config_sha256": canonical_sha256(config.raw),
        "precision": config.precision,
        "shape_profile": {
            "min": list(config.image_profile.minimum),
            "opt": list(config.image_profile.optimum),
            "max": list(config.image_profile.maximum),
        },
        "builder_backend": backend,
        "builder_command": command,
        "tensorrt_version": version,
        "platform": platform.platform(),
        "portable_across_gpu_architectures": False,
    }
    write_json_atomic(output.with_suffix(output.suffix + ".json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
