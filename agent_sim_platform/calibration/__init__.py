"""Calibration and benchmark alignment for agent-sim-platform."""

from .engine import CalibrationEngine
from .fitting import ConstantFitter, override_constants
from .metrics import aggregate_metric_errors, compute_errors, mape, rmse, r2
from .report import CalibrationReport, format_report
from .residual import ResidualModel
from ..data_models import CalibrationConfig

__all__ = [
    "CalibrationConfig",
    "CalibrationEngine",
    "CalibrationReport",
    "ConstantFitter",
    "ResidualModel",
    "aggregate_metric_errors",
    "compute_errors",
    "format_report",
    "mape",
    "override_constants",
    "rmse",
    "r2",
]
