"""Lightning wrapper that preserves the explicit image/state/target contract."""

from __future__ import annotations

from typing import Any

import torch

try:
    import lightning as L
except ImportError as exc:  # pragma: no cover - optional local dependency
    raise ImportError("Install AeroGuard with the 'ml' extra to use Lightning training") from exc


class AeroGuardDetectorModule(L.LightningModule):
    def __init__(
        self,
        detector: torch.nn.Module,
        *,
        backbone_lr: float = 3e-5,
        head_and_film_lr: float = 3e-4,
        weight_decay: float = 1e-4,
        freeze_visual_detector: bool = False,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.backbone_lr = backbone_lr
        self.head_and_film_lr = head_and_film_lr
        self.weight_decay = weight_decay
        self.freeze_visual_detector = freeze_visual_detector
        if freeze_visual_detector and any(
            parameter.requires_grad for parameter in detector.detector.parameters()
        ):
            raise ValueError(
                "freeze_visual_detector requires every visual-detector parameter "
                "to have requires_grad=False"
            )
        self.save_hyperparameters(ignore=("detector",))

    def _freeze_visual_running_statistics(self) -> None:
        for module in self.detector.detector.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True):
        """Keep frozen visual BatchNorm buffers fixed while training the adapter."""
        super().train(mode)
        if mode and self.freeze_visual_detector:
            self._freeze_visual_running_statistics()
        return self

    def training_step(self, batch: dict[str, Any], batch_index: int) -> torch.Tensor:
        if self.freeze_visual_detector:
            # Lightning can re-apply training mode between setup and the first batch.
            # Enforce the invariant immediately before every visual forward pass.
            self._freeze_visual_running_statistics()
        losses = self.detector(
            batch["images"],
            batch["state"],
            batch["state_mask"],
            batch["targets"],
        )
        if not isinstance(losses, dict) or not losses:
            raise RuntimeError("detector training forward must return a non-empty loss dictionary")
        total = sum(losses.values())
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite training loss at batch {batch_index}")
        self._aeroguard_last_train_step = int(self.global_step) + 1
        self._aeroguard_last_train_metrics = {
            "loss": float(total.detach().cpu()),
            **{name: float(value.detach().cpu()) for name, value in losses.items()},
        }
        self.log("train/loss", total, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["images"]))
        for name, value in losses.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, batch_size=len(batch["images"]))
        return total

    def configure_optimizers(self):
        backbone = [
            parameter
            for parameter in self.detector.detector.backbone.parameters()
            if parameter.requires_grad
        ]
        backbone_ids = {id(parameter) for parameter in backbone}
        other = [
            parameter
            for parameter in self.detector.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        parameter_groups = []
        if backbone:
            parameter_groups.append({"params": backbone, "lr": self.backbone_lr})
        if other:
            parameter_groups.append({"params": other, "lr": self.head_and_film_lr})
        if not parameter_groups:
            raise RuntimeError("detector has no trainable parameters")
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(self.trainer.estimated_stepping_batches)),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
