"""Protocol-aware evaluation, calibration, and report helpers."""

from .calibration import CalibrationSummary, ReliabilityBin, evaluate_calibration
from .matching import MatchSummary, match_detections
from .metrics import EvaluationResult, evaluate_detections
from .operating import evaluate_operating_points, select_best_f1_point
from .predictions import normalize_model_prediction
from .report import build_evaluation_report, write_evaluation_report
from .visdrone import (
    VISDRONE_TO_AUAIR,
    VisDroneRow,
    match_visdrone_detections,
    parse_visdrone_det_row,
)

__all__ = [
    "VISDRONE_TO_AUAIR",
    "CalibrationSummary",
    "EvaluationResult",
    "MatchSummary",
    "ReliabilityBin",
    "VisDroneRow",
    "build_evaluation_report",
    "evaluate_calibration",
    "evaluate_detections",
    "evaluate_operating_points",
    "match_detections",
    "match_visdrone_detections",
    "normalize_model_prediction",
    "parse_visdrone_det_row",
    "select_best_f1_point",
    "write_evaluation_report",
]
