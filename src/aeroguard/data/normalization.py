"""Training-only normalization for the canonical AU-AIR state vector."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .auair import STATE_FEATURE_NAMES, state_vector


@dataclass
class StateNormalizer:
    feature_names: tuple[str, ...] = STATE_FEATURE_NAMES
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    @property
    def mean(self) -> np.ndarray:
        self._check_fitted()
        return self.mean_.copy()  # type: ignore[union-attr]

    @property
    def std(self) -> np.ndarray:
        self._check_fitted()
        return self.scale_.copy()  # type: ignore[union-attr]

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StateNormalizer must be fitted before transform")

    @staticmethod
    def _matrix(states: Any) -> tuple[np.ndarray, bool]:
        if isinstance(states, Mapping):
            states = list(states.values())
        if isinstance(states, np.ndarray):
            array = np.asarray(states, dtype=np.float64)
            was_vector = array.ndim == 1
            if was_vector:
                array = array.reshape(1, -1)
            return array, was_vector
        array = np.asarray([state_vector(value) for value in states], dtype=np.float64)
        return array, False

    def _select_training_values(
        self,
        states: Mapping[Any, Any] | Sequence[Any] | np.ndarray,
        train_ids: Iterable[Any] | None,
        sample_ids: Sequence[Any] | None,
    ) -> list[Any]:
        if isinstance(states, Mapping):
            if train_ids is None:
                selected_ids = list(states.keys())
            else:
                selected_ids = list(train_ids)
            missing = [key for key in selected_ids if key not in states]
            if missing:
                raise KeyError(f"training IDs missing from state mapping: {missing[:3]}")
            return [states[key] for key in selected_ids]
        values = list(states)
        if train_ids is None:
            return values
        ids = list(train_ids)
        if ids and all(isinstance(value, (bool, np.bool_)) for value in ids):
            if len(ids) != len(values):
                raise ValueError("boolean training mask length does not match states")
            return [value for value, keep in zip(values, ids) if keep]
        if sample_ids is not None:
            allowed = set(ids)
            if len(sample_ids) != len(values):
                raise ValueError("sample_ids length does not match states")
            return [value for value, sample_id in zip(values, sample_ids) if sample_id in allowed]
        if all(isinstance(value, (int, np.integer)) for value in ids):
            return [values[int(index)] for index in ids]
        raise ValueError("non-index training IDs require a mapping or sample_ids")

    def fit(
        self,
        states: Mapping[Any, Any] | Sequence[Any] | np.ndarray,
        train_ids: Iterable[Any] | None = None,
        *,
        sample_ids: Sequence[Any] | None = None,
    ) -> StateNormalizer:
        """Fit statistics from the supplied training subset only.

        Nonfinite values are ignored per feature.  A feature with no finite
        training observations gets neutral statistics (mean 0, scale 1).
        """

        selected = self._select_training_values(states, train_ids, sample_ids)
        if not selected:
            raise ValueError("cannot fit a state normalizer with no training states")
        matrix, _ = self._matrix(selected)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("state vectors do not match the canonical feature dimension")
        finite = np.isfinite(matrix)
        means = np.zeros(matrix.shape[1], dtype=np.float64)
        scales = np.ones(matrix.shape[1], dtype=np.float64)
        for column in range(matrix.shape[1]):
            values = matrix[finite[:, column], column]
            if values.size:
                means[column] = values.mean()
                spread = values.std()
                scales[column] = spread if np.isfinite(spread) and spread > 0 else 1.0
        self.mean_ = means
        self.scale_ = scales
        return self

    def transform(self, states: Any) -> np.ndarray:
        self._check_fitted()
        array, was_vector = self._matrix(states)
        if array.ndim != 2 or array.shape[1] != len(self.feature_names):
            raise ValueError("state vectors do not match the canonical feature dimension")
        safe = np.where(np.isfinite(array), array, self.mean_)
        result = (safe - self.mean_) / self.scale_
        return result[0] if was_vector else result

    def fit_transform(self, states: Any, train_ids: Iterable[Any] | None = None, *, sample_ids: Sequence[Any] | None = None) -> np.ndarray:
        return self.fit(states, train_ids, sample_ids=sample_ids).transform(states)

    def availability_mask(self, states: Any) -> np.ndarray:
        values, _ = self._matrix(states)
        return np.isfinite(values).all(axis=1)

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {"feature_names": list(self.feature_names), "mean": self.mean.tolist(), "scale": self.std.tolist()}
