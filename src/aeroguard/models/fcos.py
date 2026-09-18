"""Explicit flight-state-conditioned TorchVision FCOS adapter."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FCOS, fcos_resnet50_fpn

from .film import GatedResidualFiLM


class FlightAwareFCOS(nn.Module):
    """FCOS with a stateless, per-sample FiLM transform on FPN features.

    The implementation mirrors TorchVision's public FCOS forward path so state
    remains an explicit input. It deliberately avoids hooks and mutable batch
    globals, which can silently break image/state pairing.
    """

    def __init__(
        self,
        detector: FCOS,
        *,
        state_dim: int = 8,
        feature_channels: int = 256,
        feature_levels: int = 5,
        hidden_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.detector = detector
        self.film = GatedResidualFiLM(
            [feature_channels] * feature_levels,
            state_dim=state_dim,
            hidden_dims=hidden_dims,
        )

    def forward(
        self,
        images: list[Tensor],
        state: Tensor,
        state_mask: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        if len(images) == 0:
            raise ValueError("images cannot be empty")
        if state.ndim != 2 or state.shape[0] != len(images):
            raise ValueError("state must have shape [number_of_images, state_dim]")
        if state_mask.shape not in {(len(images),), (len(images), 1)}:
            raise ValueError("state_mask must have shape [number_of_images] or [number_of_images, 1]")
        if self.training:
            if targets is None:
                raise ValueError("targets are required while training")
            for target in targets:
                boxes = target["boxes"]
                if not isinstance(boxes, Tensor) or boxes.ndim != 2 or boxes.shape[-1] != 4:
                    raise TypeError("target boxes must be tensors with shape [N, 4]")

        original_sizes: list[tuple[int, int]] = []
        for image in images:
            if image.ndim != 3:
                raise ValueError("each image must have shape [C, H, W]")
            original_sizes.append((int(image.shape[-2]), int(image.shape[-1])))

        transformed_images, transformed_targets = self.detector.transform(images, targets)
        if transformed_targets is not None:
            for target_index, target in enumerate(transformed_targets):
                boxes = target["boxes"]
                degenerate = boxes[:, 2:] <= boxes[:, :2]
                if degenerate.any():
                    first = int(torch.where(degenerate.any(dim=1))[0][0])
                    raise ValueError(
                        f"target {target_index} has a degenerate box after transform: "
                        f"{boxes[first].tolist()}"
                    )

        feature_map = self.detector.backbone(transformed_images.tensors)
        if isinstance(feature_map, Tensor):
            feature_map = OrderedDict([("0", feature_map)])
        features = list(feature_map.values())
        conditioned = self.film(features, state, state_mask)
        if not isinstance(conditioned, tuple):
            raise TypeError("feature-pyramid conditioning must return a tuple")
        conditioned_features = list(conditioned)

        head_outputs = self.detector.head(conditioned_features)
        anchors = self.detector.anchor_generator(transformed_images, conditioned_features)
        anchors_per_level = [feature.size(2) * feature.size(3) for feature in conditioned_features]

        if self.training:
            assert transformed_targets is not None
            return self.detector.compute_loss(
                transformed_targets,
                head_outputs,
                anchors,
                anchors_per_level,
            )

        split_outputs = {
            key: list(value.split(anchors_per_level, dim=1)) for key, value in head_outputs.items()
        }
        split_anchors = [list(item.split(anchors_per_level)) for item in anchors]
        detections = self.detector.postprocess_detections(
            split_outputs,
            split_anchors,
            transformed_images.image_sizes,
        )
        return self.detector.transform.postprocess(
            detections,
            transformed_images.image_sizes,
            original_sizes,
        )


def build_flight_aware_fcos(
    *,
    pretrained: bool = False,
    min_size: int = 540,
    max_size: int = 960,
    state_dim: int = 8,
) -> FlightAwareFCOS:
    """Build the nine-class AU-AIR detector with an explicit weight policy."""

    if pretrained:
        warnings.warn(
            "COCO detector weights are not loaded automatically because replacing the "
            "classification predictor requires recording an approved origin and checksum. "
            "Only ImageNet backbone initialization is used by this helper.",
            stacklevel=2,
        )
    detector = fcos_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT if pretrained else None,
        num_classes=9,
        min_size=min_size,
        max_size=max_size,
    )
    return FlightAwareFCOS(detector, state_dim=state_dim)


__all__ = ["FlightAwareFCOS", "build_flight_aware_fcos"]
