"""Golden-file regression tests for calibration and profiling outputs."""

import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


def test_golden_calibration_capacity_exists():
    path = GOLDEN_DIR / "calibrate_capacity.json"
    assert path.exists()


def test_golden_calibration_capacity_has_expected_keys():
    path = GOLDEN_DIR / "calibrate_capacity.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "fixture_results" in data
    assert "metric_summary" in data
    assert "overall_mape" in data
    assert "fitted_values" in data


def test_golden_profile_trace_exists():
    path = GOLDEN_DIR / "profile_trace.md"
    assert path.exists()


def test_golden_profile_trace_has_expected_sections():
    path = GOLDEN_DIR / "profile_trace.md"
    text = path.read_text(encoding="utf-8")
    assert "# Profiling Report" in text
    assert "## Software Layer" in text
    assert "## Hardware Layer" in text
    assert "## Algorithm Layer" in text
    assert "## System Layer" in text
    assert "## Recommendations" in text
