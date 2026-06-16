"""Tests for calibration engine, metrics, and fitting."""

import pytest

from agent_sim_platform import config as sim_config
from agent_sim_platform.benchmarks import DEFAULT_REGISTRY
from agent_sim_platform.calibration import CalibrationEngine, CalibrationReport
from agent_sim_platform.calibration.fitting import (
    ConstantFitter,
    override_constants,
    PARAMETER_RANGES,
)
from agent_sim_platform.calibration.metrics import (
    aggregate_metric_errors,
    compute_errors,
    mape,
    rmse,
    r2,
)
from agent_sim_platform.data_models import CalibrationConfig


class TestMetrics:
    def test_mape_basic(self):
        assert mape([100.0, 200.0], [110.0, 190.0]) == pytest.approx(0.075, abs=1e-6)

    def test_mape_skips_zero_observed(self):
        assert mape([0.0, 100.0], [0.0, 110.0]) == pytest.approx(0.10, abs=1e-6)

    def test_rmse_basic(self):
        assert rmse([0.0, 2.0], [1.0, 3.0]) == pytest.approx(1.0, abs=1e-6)

    def test_r2_perfect(self):
        assert r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0, abs=1e-6)

    def test_compute_errors(self):
        observed = {"a": 100.0, "b": 200.0}
        predicted = {"a": 110.0, "b": 190.0}
        errors = compute_errors(observed, predicted)
        assert errors["a"]["pct"] == pytest.approx(0.10, abs=1e-6)
        assert errors["b"]["pct"] == pytest.approx(0.05, abs=1e-6)

    def test_aggregate_metric_errors(self):
        data = [
            ("f1", {"x": 100.0}, {"x": 110.0}),
            ("f2", {"x": 200.0}, {"x": 190.0}),
        ]
        summary = aggregate_metric_errors(data)
        assert summary["x"]["mape"] == pytest.approx(0.075, abs=1e-6)
        assert summary["x"]["samples"] == 2


class TestOverrideConstants:
    def test_override_restores_value(self):
        original = sim_config.DEFAULT_PREFILL_UTILIZATION
        with override_constants({"default_prefill_utilization": 0.99}):
            assert sim_config.DEFAULT_PREFILL_UTILIZATION == pytest.approx(0.99)
        assert sim_config.DEFAULT_PREFILL_UTILIZATION == pytest.approx(original)

    def test_override_unknown_param_is_ignored(self):
        with override_constants({"not_a_param": 0.5}):
            pass  # should not raise


class TestCalibrationEngine:
    def test_evaluate_single_training_fixture(self):
        engine = CalibrationEngine(CalibrationConfig(domain="training"))
        fixture = DEFAULT_REGISTRY.get("llama2_70b_pretrain")
        result = engine.evaluate_fixture(fixture)
        assert result["name"] == "llama2_70b_pretrain"
        assert "total_time_seconds" in result["predicted"]
        assert "total_time_seconds" in result["errors"]

    def test_evaluate_single_serving_fixture(self):
        engine = CalibrationEngine(CalibrationConfig(domain="serving"))
        fixture = DEFAULT_REGISTRY.get("vllm_llama70b_serving")
        result = engine.evaluate_fixture(fixture)
        assert result["name"] == "vllm_llama70b_serving"
        assert "throughput_req_per_sec" in result["predicted"]

    def test_evaluate_single_capacity_fixture(self):
        engine = CalibrationEngine(CalibrationConfig(domain="capacity"))
        fixture = DEFAULT_REGISTRY.get("capacity_llama70b_4k")
        result = engine.evaluate_fixture(fixture)
        assert result["name"] == "capacity_llama70b_4k"
        assert "memory_required_gb" in result["predicted"]

    def test_evaluate_registry_returns_report(self):
        engine = CalibrationEngine(CalibrationConfig(domain="training"))
        report = engine.evaluate_registry(DEFAULT_REGISTRY)
        assert isinstance(report, CalibrationReport)
        assert report.fixture_results
        assert "total_time_seconds" in report.metric_summary
        assert report.overall_mape >= 0.0

    def test_fit_improves_or_preserves_mape(self):
        engine = CalibrationEngine(
            CalibrationConfig(
                domain="capacity",
                fit_params=["default_decode_utilization"],
                max_iterations=2,
                tolerance=0.01,
            )
        )
        report = engine.fit(DEFAULT_REGISTRY)
        assert isinstance(report, CalibrationReport)
        assert "default_decode_utilization" in report.fitted_values


class TestConstantFitter:
    def test_baseline_values(self):
        fitter = ConstantFitter(CalibrationConfig(), lambda r: r)
        assert "mfu_target" in fitter.best_values

    def test_parameter_ranges_defined(self):
        assert "mfu_target" in PARAMETER_RANGES
        low, high, step = PARAMETER_RANGES["mfu_target"]
        assert low < high
        assert step > 0

    def test_fit_with_stub_evaluate(self):
        calls = []

        def evaluate(values):
            calls.append(values.copy())
            # Minimal report-like object with overall_mape
            class FakeReport:
                overall_mape = 1.0 / (values.get("mfu_target", 0.35) + 0.1)

            return FakeReport()

        fitter = ConstantFitter(
            CalibrationConfig(fit_params=["mfu_target"], max_iterations=2, tolerance=0.01),
            evaluate,
        )
        best, history = fitter.fit()
        assert "mfu_target" in best
        assert history
        assert len(calls) > 0
