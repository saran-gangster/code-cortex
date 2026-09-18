"""Stateless gated residual FiLM conditioning for feature-pyramid tensors."""

from __future__ import annotations

from collections.abc import Sequence

try:  # Keep importing the data package possible in lightweight environments.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised only without PyTorch installed.
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class GatedResidualFiLM(nn.Module):
        """Apply per-sample state-conditioned residual affine transforms.

        For each feature map ``F`` this computes ``(1 + m*gamma) * F + m*beta``.
        The final projections for gamma/beta start at zero, making the module an
        exact identity before training.  Invalid state is sanitized before it
        reaches the MLP; a zero sample mask disables its transform entirely.
        """

        def __init__(
            self,
            feature_channels: int | Sequence[int],
            state_dim: int = 8,
            hidden_dims: Sequence[int] = (64, 64),
        ) -> None:
            super().__init__()
            if isinstance(feature_channels, int):
                channels = (feature_channels,)
            else:
                channels = tuple(int(value) for value in feature_channels)
            if not channels or any(value <= 0 for value in channels):
                raise ValueError("feature_channels must contain positive integers")
            if state_dim <= 0 or not hidden_dims or any(int(value) <= 0 for value in hidden_dims):
                raise ValueError("state_dim and hidden_dims must be positive")
            self.feature_channels = channels
            self.state_dim = int(state_dim)
            layers: list[nn.Module] = []
            input_dim = self.state_dim
            for hidden_dim in hidden_dims:
                layers.extend((nn.Linear(input_dim, int(hidden_dim)), nn.ReLU()))
                input_dim = int(hidden_dim)
            self.encoder = nn.Sequential(*layers)
            self.projections = nn.ModuleList(
                [nn.Linear(input_dim, 2 * channels_for_level) for channels_for_level in channels]
            )
            for projection in self.projections:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

        @staticmethod
        def _as_sample_mask(state_mask: Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
            if state_mask is None:
                return torch.ones(batch_size, device=device, dtype=dtype)
            mask = torch.as_tensor(state_mask, device=device, dtype=dtype)
            if mask.ndim == 0:
                mask = mask.expand(batch_size)
            elif mask.ndim == 2 and mask.shape[1] == 1:
                mask = mask[:, 0]
            elif mask.ndim != 1:
                raise ValueError("state_mask must have shape [B] or [B, 1]")
            if mask.shape[0] != batch_size:
                raise ValueError("state_mask batch dimension does not match features")
            return torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        def forward(
            self,
            features: Tensor | Sequence[Tensor],
            state: Tensor,
            state_mask: Tensor | None = None,
        ) -> Tensor | tuple[Tensor, ...]:
            feature_list = (features,) if isinstance(features, Tensor) else tuple(features)
            if len(feature_list) != len(self.projections):
                raise ValueError(
                    f"expected {len(self.projections)} feature levels, got {len(feature_list)}"
                )
            if not feature_list or any(feature.ndim < 2 for feature in feature_list):
                raise ValueError("feature maps must have a batch and channel dimension")
            batch_size = feature_list[0].shape[0]
            if any(feature.shape[0] != batch_size for feature in feature_list):
                raise ValueError("all feature levels must share the batch dimension")
            reference = feature_list[0]
            safe_state = torch.as_tensor(state, device=reference.device, dtype=reference.dtype)
            if safe_state.ndim != 2 or safe_state.shape != (batch_size, self.state_dim):
                raise ValueError(
                    f"state must have shape ({batch_size}, {self.state_dim}), got {tuple(safe_state.shape)}"
                )
            # This must happen before the MLP: NaN * zero is still NaN.
            safe_state = torch.nan_to_num(safe_state, nan=0.0, posinf=0.0, neginf=0.0)
            mask = self._as_sample_mask(state_mask, batch_size, reference.device, reference.dtype)
            encoded = self.encoder(safe_state)
            outputs: list[Tensor] = []
            for feature, projection, expected_channels in zip(feature_list, self.projections, self.feature_channels):
                if feature.shape[1] != expected_channels:
                    raise ValueError(
                        f"feature has {feature.shape[1]} channels; expected {expected_channels}"
                    )
                parameters = projection(encoded)
                gamma, beta = parameters.chunk(2, dim=1)
                view_shape = (batch_size, expected_channels) + (1,) * (feature.ndim - 2)
                gate = mask.reshape(batch_size, 1, *([1] * (feature.ndim - 2)))
                gamma = gamma.reshape(view_shape)
                beta = beta.reshape(view_shape)
                outputs.append((1.0 + gate * gamma) * feature + gate * beta)
            if isinstance(features, Tensor):
                return outputs[0]
            return tuple(outputs)


else:

    class GatedResidualFiLM:  # type: ignore[no-redef]
        """Placeholder that gives an actionable error when torch is absent."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("GatedResidualFiLM requires PyTorch")


StateFiLM = GatedResidualFiLM

__all__ = ["GatedResidualFiLM", "StateFiLM"]
