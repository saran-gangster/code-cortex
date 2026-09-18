"""Reusable model components; the full FCOS detector remains an integration concern."""

from .film import GatedResidualFiLM, StateFiLM

try:
    from .fcos import FlightAwareFCOS, build_flight_aware_fcos
except ImportError:  # TorchVision is an optional ML dependency.
    FlightAwareFCOS = None  # type: ignore[assignment]
    build_flight_aware_fcos = None  # type: ignore[assignment]

__all__ = [
    "FlightAwareFCOS",
    "GatedResidualFiLM",
    "StateFiLM",
    "build_flight_aware_fcos",
]
