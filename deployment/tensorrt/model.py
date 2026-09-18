"""Lazy PyTorch model loading and raw pre-NMS FCOS export wrapper."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .config import DeploymentConfig
from .metadata import sha256_file
from .postprocess import LEVELS, output_names


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("PyTorch is required; install AeroGuard with the 'ml' extra") from exc
    return torch


def verify_checkpoint_provenance(
    checkpoint: Path,
    run_summary: Path | None,
    *,
    allow_unverified: bool,
) -> tuple[str, dict[str, Any] | None]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".ckpt":
        raise ValueError("checkpoint must be a regular .ckpt file from AeroGuard Lightning training")
    checkpoint_sha = sha256_file(checkpoint)
    inferred = checkpoint.parent.parent / "run_summary.json" if checkpoint.parent.name == "checkpoints" else checkpoint.parent / "run_summary.json"
    summary_path = run_summary.resolve() if run_summary else inferred
    if not summary_path.is_file():
        if allow_unverified:
            return checkpoint_sha, None
        raise RuntimeError(
            f"checkpoint provenance summary is missing: {summary_path}; "
            "use --allow-unverified-provenance only for a trusted local checkpoint"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    relative = checkpoint.relative_to(summary_path.parent).as_posix()
    expected = summary.get("checkpoint_sha256s", {}).get(relative)
    if expected is None and checkpoint.name == "final.ckpt":
        expected = summary.get("checkpoint_sha256")
    if expected != checkpoint_sha:
        raise RuntimeError(
            f"checkpoint SHA-256 does not match run summary: {checkpoint_sha} != {expected!r}"
        )
    if summary.get("final_test_unsealed") is not False:
        raise RuntimeError("checkpoint summary does not preserve the final-test seal")
    return checkpoint_sha, summary


def load_flight_aware_model(checkpoint: Path, config: DeploymentConfig):
    torch = _torch()
    try:
        from aeroguard.models.fcos import build_flight_aware_fcos
    except ImportError as exc:
        raise RuntimeError("torchvision detection is required to export AeroGuard FCOS") from exc

    model = build_flight_aware_fcos(
        pretrained=False,
        min_size=config.min_size,
        max_size=config.max_size,
        state_dim=config.state_dim,
    )
    # weights_only constrains deserialization to tensor-oriented safe globals. The CLI also
    # accepts only the project's trusted Lightning .ckpt convention.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise TypeError("checkpoint must contain the AeroGuard Lightning state_dict mapping")
    wrapped = payload["state_dict"]
    if not wrapped or not all(isinstance(key, str) and key.startswith("detector.") for key in wrapped):
        raise RuntimeError("checkpoint state_dict does not use the AeroGuard detector.* key convention")
    state_dict = {key.removeprefix("detector."): value for key, value in wrapped.items()}
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise TypeError("checkpoint state_dict contains non-tensor values")
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def create_raw_export_model(model):
    """Return an nn.Module producing per-level logits/regression/centerness/anchors."""

    torch = _torch()

    class RawFCOS(torch.nn.Module):
        def __init__(self, flight_model) -> None:
            super().__init__()
            self.flight_model = flight_model
            generator = flight_model.detector.anchor_generator
            counts = generator.num_anchors_per_location()
            if counts != [1] * len(LEVELS):
                raise RuntimeError(f"raw exporter requires one FCOS anchor per location, got {counts}")
            if len(generator.cell_anchors) != len(LEVELS):
                raise RuntimeError("anchor generator does not have five FCOS levels")
            for index, anchor in enumerate(generator.cell_anchors):
                self.register_buffer(f"base_anchor_{index}", anchor.detach().clone())

        def _anchors(self, feature, image, index: int):
            height, width = feature.shape[-2], feature.shape[-1]
            stride_height = image.shape[-2] // height
            stride_width = image.shape[-1] // width
            shifts_x = torch.arange(width, dtype=torch.int32, device=feature.device) * stride_width
            shifts_y = torch.arange(height, dtype=torch.int32, device=feature.device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack(
                (shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)),
                dim=1,
            ).to(feature.dtype)
            base = getattr(self, f"base_anchor_{index}").to(feature.dtype)
            return (shifts[:, None, :] + base[None, :, :]).reshape(1, -1, 4)

        def forward(self, images, state, state_mask):
            feature_map = self.flight_model.detector.backbone(images)
            if isinstance(feature_map, torch.Tensor):
                feature_map = OrderedDict((("0", feature_map),))
            features = list(feature_map.values())
            conditioned = self.flight_model.film(features, state, state_mask)
            results = []
            for index, feature in enumerate(conditioned):
                head = self.flight_model.detector.head([feature])
                results.extend(
                    (
                        head["cls_logits"],
                        head["bbox_regression"],
                        head["bbox_ctrness"],
                        self._anchors(feature, images, index),
                    )
                )
            return tuple(results)

    wrapper = RawFCOS(model).eval()
    if len(output_names()) != 20:
        raise AssertionError("unexpected raw output contract")
    return wrapper


__all__ = [
    "create_raw_export_model",
    "load_flight_aware_model",
    "verify_checkpoint_provenance",
]
