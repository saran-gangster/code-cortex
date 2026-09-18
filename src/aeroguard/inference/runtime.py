"""Small runtime boundary: real model code can be injected without changing the API."""

from __future__ import annotations

from collections.abc import Callable

from .contracts import InferenceRecord, InferenceRequest, PredictionSource


class RuntimeErrorInfo(RuntimeError):
    """A model/runtime failure that must be surfaced as an error, never as detections."""


class InferenceRuntime:
    def __init__(self, predict: Callable[[InferenceRequest], InferenceRecord] | None = None) -> None:
        self._predict = predict
        self.ready = predict is not None

    def infer(self, request: InferenceRequest) -> InferenceRecord:
        if self._predict is None:
            raise RuntimeErrorInfo("computed inference runtime is not configured")
        try:
            record = self._predict(request)
        except RuntimeErrorInfo:
            raise
        except Exception as exc:
            raise RuntimeErrorInfo(f"inference failed: {exc}") from exc
        if not isinstance(record, InferenceRecord):
            raise RuntimeErrorInfo("inference runtime returned an invalid record")
        try:
            record = InferenceRecord.model_validate(record.model_dump())
        except Exception as exc:
            raise RuntimeErrorInfo("inference runtime returned an invalid record") from exc
        if record.prediction_source is not PredictionSource.computed:
            raise RuntimeErrorInfo("injected runtime must return computed provenance")
        if record.frame_id != request.frame_id:
            raise RuntimeErrorInfo("inference runtime returned a record for a different frame")
        if record.original_size != request.original_size:
            raise RuntimeErrorInfo("inference runtime returned a record for a different image size")
        if record.input_mode is not request.input_mode:
            raise RuntimeErrorInfo("inference runtime returned a record for a different input mode")
        if record.metadata_alignment is not request.metadata_alignment:
            raise RuntimeErrorInfo("inference runtime returned a record for different metadata alignment")
        return record
