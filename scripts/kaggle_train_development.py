#!/usr/bin/env python3
"""Train one matched AeroGuard development arm on frozen AU-AIR train roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor

from aeroguard.data.provenance import validate_protocol_bundle
from aeroguard.models.fcos import build_flight_aware_fcos
from aeroguard.training import AeroGuardDetectorModule


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_training_records(manifest: Path, protocol: dict, *, seed: int) -> list[dict]:
    train_roots = set(protocol["selection"]["train"])
    records: list[dict] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["recording_root"] in train_roots:
                records.append(record)
    if not records:
        raise ValueError("manifest contains no records from the frozen training roots")
    records.sort(key=lambda item: item["frame_id"])
    random.Random(seed).shuffle(records)
    return records


class DevelopmentScheduleDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        records: list[dict],
        *,
        steps: int,
        start_step: int,
        state_mask: float,
        means: list[float],
        scales: list[float],
    ) -> None:
        self.data_root = data_root
        self.records = records
        self.steps = steps
        self.start_step = start_step
        self.state_mask = state_mask
        self.means = torch.tensor(means, dtype=torch.float32)
        self.scales = torch.tensor(scales, dtype=torch.float32)
        if self.means.shape != (8,) or self.scales.shape != (8,):
            raise ValueError("state normalizer must contain eight means and scales")
        if not torch.isfinite(self.means).all() or not torch.isfinite(self.scales).all():
            raise ValueError("state normalizer must be finite")
        if not torch.all(self.scales > 0):
            raise ValueError("state normalizer scales must be positive")

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int):
        record = self.records[(self.start_step + index) % len(self.records)]
        with Image.open(self.data_root / record["image_relative_path"]) as image:
            pixels = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
        state = torch.tensor(record["state"], dtype=torch.float32)
        state = (state - self.means) / self.scales
        if not torch.isfinite(state).all():
            raise ValueError(f"non-finite normalized state for {record['frame_id']}")
        target = {
            "boxes": torch.tensor(record["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(record["model_labels"], dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return (
            pixels,
            target,
            state,
            torch.tensor(self.state_mask, dtype=torch.float32),
        )


def collate(samples):
    images, targets, state, mask = zip(*samples)
    return {
        "images": list(images),
        "targets": list(targets),
        "state": torch.stack(state),
        "state_mask": torch.stack(mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/auair/04_AUAIR_multimodal_uav"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/auair.jsonl"))
    parser.add_argument("--protocol", type=Path, default=Path("manifests/protocol.json"))
    parser.add_argument("--normalizer", type=Path, default=Path("manifests/state_normalizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument(
        "--warmstart-origin",
        default="random_no_pretrained_weights",
        help="Human-readable, disclosed origin shared by both matched arms.",
    )
    parser.add_argument("--mode", choices=("masked", "paired"), required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", choices=("32-true", "16-mixed"), default="32-true")
    args = parser.parse_args()
    if args.steps <= 0 or args.checkpoint_steps <= 0 or args.num_workers < 0:
        raise ValueError("steps/checkpoint-steps must be positive and num-workers non-negative")

    args.output.mkdir(parents=True, exist_ok=True)
    L.seed_everything(args.seed, workers=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    normalizer = json.loads(args.normalizer.read_text(encoding="utf-8"))
    partitions = validate_protocol_bundle(protocol, normalizer)

    records = load_training_records(args.manifest, protocol, seed=args.seed)
    used_roots = {record["recording_root"] for record in records}
    if used_roots != set(partitions.train):
        raise RuntimeError("training manifest roots do not exactly match the frozen train partition")
    resume_checkpoint = args.output / "checkpoints" / "last.ckpt"
    start_step = 0
    if resume_checkpoint.exists():
        resume_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
        start_step = int(resume_payload.get("global_step", -1))
        if not 0 <= start_step < args.steps:
            raise RuntimeError(f"invalid resume global_step {start_step} for {args.steps} requested steps")
    schedule = [records[index % len(records)]["frame_id"] for index in range(args.steps)]
    state_mask = 1.0 if args.mode == "paired" else 0.0
    dataset = DevelopmentScheduleDataset(
        args.data_root,
        records,
        steps=args.steps - start_step,
        start_step=start_step,
        state_mask=state_mask,
        means=normalizer["mean"],
        scales=normalizer["std"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate,
    )

    detector = build_flight_aware_fcos(pretrained=False, min_size=320, max_size=576)
    detector.load_state_dict(torch.load(args.warmstart, map_location="cpu", weights_only=True))
    module = AeroGuardDetectorModule(detector)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output / "checkpoints",
        filename="step-{step:06d}",
        every_n_train_steps=args.checkpoint_steps,
        save_top_k=-1,
        save_last=True,
        save_weights_only=False,
    )
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        max_steps=args.steps,
        max_epochs=1,
        logger=False,
        callbacks=[checkpoint_callback],
        enable_model_summary=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=20,
        deterministic=True,
    )
    trainer.fit(
        module,
        train_dataloaders=loader,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint.exists() else None,
    )
    if int(trainer.global_step) != args.steps:
        raise RuntimeError(
            f"training stopped at global step {trainer.global_step}; expected exactly {args.steps}"
        )
    final_checkpoint = args.output / "final.ckpt"
    trainer.save_checkpoint(final_checkpoint)
    final_loss = trainer.callback_metrics.get("train/loss_step")
    summary = {
        "artifact_kind": "matched_development_training_not_benchmark",
        "benchmark_claim": False,
        "experiment_arm": "E2_paired_film" if args.mode == "paired" else "E1_rgb_masked",
        "completed_steps": int(trainer.global_step),
        "requested_steps": args.steps,
        "resumed_from_step": start_step,
        "training_record_count": len(records),
        "training_roots": protocol["selection"]["train"],
        "development_roots_used": sorted(used_roots & set(partitions.development)),
        "final_test_roots_used": sorted(used_roots & set(partitions.final_test)),
        "final_test_unsealed": False,
        "matched_frame_schedule_sha256": canonical_hash(schedule),
        "state_alignment": "paired_annotation" if args.mode == "paired" else "unavailable",
        "state_mask": state_mask,
        "normalizer_sha256": normalizer["normalizer_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "initial_weights_origin": args.warmstart_origin,
        "shared_warmstart_sha256": sha256_file(args.warmstart),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "final_logged_train_loss": float(final_loss) if final_loss is not None else None,
        "precision": args.precision,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "physical_gpu_id": os.getenv("AEROGUARD_PHYSICAL_GPU", "unrecorded"),
        "lightning_version": L.__version__,
        "torch_version": torch.__version__,
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
