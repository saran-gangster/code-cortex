"""Deterministic recording-root grouped protocol construction."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .auair import FrameRecord, recording_root


class GroupedProtocol(dict):
    """Dictionary protocol with attribute access for ergonomic callers."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _record_id(record: Any, index: int) -> str:
    return str(_field(record, "frame_id", _field(record, "image_name", index)))


def build_grouped_protocol(
    records: Iterable[FrameRecord | Mapping[str, Any]],
    train_groups: int = 5,
    development_groups: int = 1,
    final_test_groups: int = 2,
    seed: int = 17,
    *,
    group_counts: Mapping[str, int] | None = None,
) -> GroupedProtocol:
    """Assign complete recording roots to disjoint deterministic partitions.

    Group order is derived from a SHA-256 hash of ``seed`` and root ID, so the
    result is independent of input order and Python's process hash seed.  No
    image-level random split is possible through this API.
    """

    if group_counts is None:
        group_counts = {
            "train": int(train_groups),
            "development": int(development_groups),
            "final_test": int(final_test_groups),
        }
    else:
        group_counts = {str(name): int(count) for name, count in group_counts.items()}
    if any(count < 0 for count in group_counts.values()):
        raise ValueError("group counts must be non-negative")
    if not group_counts:
        raise ValueError("at least one partition is required")

    materialized = list(records)
    roots = sorted({recording_root(_field(record, "image_name", record)) for record in materialized})
    requested = sum(group_counts.values())
    if requested != len(roots):
        raise ValueError(
            f"group counts sum to {requested}, but {len(roots)} recording roots were found; "
            "make every root belong to exactly one declared partition"
        )
    ranked_roots = sorted(
        roots,
        key=lambda root: hashlib.sha256(f"{seed}:{root}".encode()).hexdigest(),
    )
    partitions: dict[str, list[str]] = {}
    cursor = 0
    for name, count in group_counts.items():
        partitions[name] = ranked_roots[cursor : cursor + count]
        cursor += count
    root_to_partition = {
        root: partition for partition, assigned in partitions.items() for root in assigned
    }
    if len(root_to_partition) != len(roots):
        raise AssertionError("internal protocol error: a root was assigned more than once")

    record_to_partition: dict[str, str] = {}
    frame_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for index, record in enumerate(materialized):
        root = recording_root(_field(record, "image_name", record))
        partition = root_to_partition[root]
        record_id = _record_id(record, index)
        if record_id in record_to_partition and record_to_partition[record_id] != partition:
            raise AssertionError(f"record identity appears in multiple partitions: {record_id}")
        record_to_partition[record_id] = partition
        frame_counts[partition] += 1
        labels = _field(record, "labels", ())
        if not labels and isinstance(record, Mapping):
            labels = _field(record, "model_labels", ())
        class_counts[partition].update(int(label) for label in labels)

    return GroupedProtocol(
        seed=int(seed),
        group_key="recording_root",
        group_counts=dict(group_counts),
        partitions=partitions,
        root_to_partition=root_to_partition,
        record_to_partition=record_to_partition,
        frame_counts=dict(frame_counts),
        class_counts={name: dict(counts) for name, counts in class_counts.items()},
    )


def assert_protocol_is_leakage_safe(protocol: Mapping[str, Any]) -> None:
    partitions = protocol["partitions"]
    seen: dict[str, str] = {}
    for partition, roots in partitions.items():
        for root in roots:
            if root in seen:
                raise AssertionError(f"recording root {root} appears in {seen[root]} and {partition}")
            seen[root] = partition
    if set(seen) != set(protocol["root_to_partition"]):
        raise AssertionError("root-to-partition mapping is incomplete")


make_grouped_protocol = build_grouped_protocol
