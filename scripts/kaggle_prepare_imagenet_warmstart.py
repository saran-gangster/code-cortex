#!/usr/bin/env python3
"""Create the disclosed shared ImageNet-backbone initialization for fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights

from aeroguard.models.fcos import build_flight_aware_fcos


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/shared_fcos_film_imagenet_init.pt"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("checkpoints/shared_fcos_film_imagenet_init.json"),
    )
    args = parser.parse_args()

    torch.manual_seed(17)
    model = build_flight_aware_fcos(pretrained=True, min_size=320, max_size=576)
    projection_l1 = sum(
        parameter.detach().abs().sum().item()
        for projection in model.film.projections
        for parameter in projection.parameters()
    )
    if projection_l1 != 0.0:
        raise RuntimeError("shared FiLM projections must start as an exact visual identity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    payload = {
        "artifact_kind": "shared_finetuning_warmstart",
        "benchmark_claim": False,
        "weights_origin": "torchvision_resnet50_imagenet1k_v2_backbone",
        "torchvision_weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "upstream_url": ResNet50_Weights.IMAGENET1K_V2.url,
        "detector_head_origin": "seeded_random_auair_9_class_head",
        "film_origin": "seeded_zero_initialized_identity",
        "seed": 17,
        "checkpoint_sha256": sha256_file(args.output),
    }
    args.summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
