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
    ) -> None:
        super().__init__()
        self.detector = detector
        self.backbone_lr = backbone_lr
        self.head_and_film_lr = head_and_film_lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=("detector",))

    def training_step(self, batch: dict[str, Any], batch_index: int) -> torch.Tensor:
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
        self.log("train/loss", total, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["images"]))
        for name, value in losses.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, batch_size=len(batch["images"]))
        return total

    def configure_optimizers(self):
        backbone = list(self.detector.detector.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone}
        other = [parameter for parameter in self.detector.parameters() if id(parameter) not in backbone_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": self.backbone_lr},
                {"params": other, "lr": self.head_and_film_lr},
            ],
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(self.trainer.estimated_stepping_batches)),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
