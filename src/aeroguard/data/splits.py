"""Compatibility imports for the frozen grouped evaluation protocol."""

from .protocol import (
    GroupedProtocol,
    assert_protocol_is_leakage_safe,
    build_grouped_protocol,
    make_grouped_protocol,
)

__all__ = ["GroupedProtocol", "assert_protocol_is_leakage_safe", "build_grouped_protocol", "make_grouped_protocol"]
