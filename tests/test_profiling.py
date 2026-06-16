"""Tests for multi-layer profiling."""

from pathlib import Path

import pytest

from agent_sim_platform.data_models import (
    AgentHarnessSpec,
    HardwareSpec,
    ModelSpec,
    OptimizationConfig,
    SimulationConfig,
    SimulationResult,
    WorkloadSpec,
)
from agent_sim_platform.profiling import ProfilingOrchestrator, TraceAnalyzer
from agent_sim_platform.profiling.layers import (
    AlgorithmLayerProfiler,
    HardwareLayerProfiler,
    SoftwareLayerProfiler,
    SystemLayerProfiler,
)


@pytest.fixture
def sample_result():
    model = ModelSpec(
        name="70B-Dense",
        total_params_b=70,
        active_params_b=70,
        n_layers=80,
        d_model=8192,
        n_heads=64,
        d_head=128,
    )
    hardware = HardwareSpec(
        name="H100-SXM5",
        vendor="nvidia",
        kind="gpu",
        memory_gb=80.0,
        memory_bw_tb_s=3.35,
        fp16_tflops=989.0,
        fp8_tflops=1979.0,
    )
    config = SimulationConfig(
        hardware=hardware,
        model=model,
        workload=WorkloadSpec(name="test", max_steps=1, avg_steps=1.0, step_std=0.0, context_limit=32768),
        harness=AgentHarnessSpec(name="test", concurrency=1),
        precision="FP8",
    )
    return SimulationResult(
        config=config,
        latency_seconds=10.0,
        wall_time_seconds=10.0,
        tokens_total=1000,
        tokens_input=800,
        tokens_output=200,
        peak_kv_gb=5.0,
        memory_required_gb=40.0,
        gpu_count=1,
        feasible=True,
        bottleneck="compute",
        cost_usd=0.01,
        utilization_gpu=0.5,
        metadata={"total_flops": 1e15},
        per_step=[
            {
                "tool": "view",
                "prefill_input": 500,
                "decode_output": 100,
                "prefill_time_ms": 30.0,
                "decode_time_ms": 400.0,
                "tool_time_ms": 20.0,
            },
            {
                "tool": "edit",
                "prefill_input": 300,
                "decode_output": 100,
                "prefill_time_ms": 20.0,
                "decode_time_ms": 400.0,
                "tool_time_ms": 40.0,
            },
        ],
    )


class TestSoftwareLayerProfiler:
    def test_profile_simulation(self, sample_result):
        profiler = SoftwareLayerProfiler()
        profile = profiler.profile_simulation(sample_result)
        assert profile["samples"] == 2
        assert profile["tool_counts"]["view"] == 1
        assert profile["tool_counts"]["edit"] == 1
        assert profile["total_prefill_tokens"] == 800
        assert profile["total_decode_tokens"] == 200
        assert profile["tool_time_fraction"] > 0.0

    def test_empty_per_step(self):
        result = SimulationResult(
            config=None,
            per_step=[],
        )
        profile = SoftwareLayerProfiler().profile_simulation(result)
        assert profile["samples"] == 0


class TestHardwareLayerProfiler:
    def test_profile(self, sample_result):
        profiler = HardwareLayerProfiler()
        profile = profiler.profile(sample_result)
        assert profile["hardware"] == "H100-SXM5"
        assert profile["gpu_count"] == 1
        assert profile["utilization_gpu"] == 0.5
        assert profile["memory_utilization"] == 0.5
        assert profile["cost_per_million_tokens"] == pytest.approx(10.0, rel=1e-3)


class TestAlgorithmLayerProfiler:
    def test_profile(self, sample_result):
        profiler = AlgorithmLayerProfiler()
        profile = profiler.profile(sample_result.config.model, "FP8")
        assert profile["algorithm_family"] == "dense"
        assert profile["total_params_b"] == 70
        assert profile["active_params_b"] == 70
        assert profile["flops_per_token_forward"] == pytest.approx(140e9, abs=1e6)
        assert profile["kv_bytes_per_token"] > 0


class TestSystemLayerProfiler:
    def test_profile(self, sample_result):
        profiler = SystemLayerProfiler()
        profile = profiler.profile(sample_result)
        assert profile["harness"] == "test"
        assert profile["concurrency"] == 1
        assert profile["bottleneck"] == "compute"


class TestProfilingOrchestrator:
    def test_profile_simulation(self, sample_result):
        orchestrator = ProfilingOrchestrator()
        report = orchestrator.profile_simulation(sample_result)
        assert set(report.layers.keys()) == {"software", "hardware", "algorithm", "system"}
        assert report.correlations
        assert report.recommendations
        assert report.raw_samples

    def test_profile_trace(self):
        trace_path = Path(__file__).parent / "fixtures" / "trace.jsonl"
        orchestrator = ProfilingOrchestrator()
        report = orchestrator.profile_trace(str(trace_path))
        assert set(report.layers.keys()) == {"software", "hardware", "algorithm", "system"}
        assert report.layers["software"]["samples"] == 2

    def test_compare_simulation_and_trace(self, sample_result):
        trace_path = Path(__file__).parent / "fixtures" / "trace.jsonl"
        orchestrator = ProfilingOrchestrator()
        comparison = orchestrator.compare(sample_result, str(trace_path))
        assert "software" in comparison
        assert "hardware" in comparison
        assert "recommendations" in comparison

    def test_recommendations_memory_pressure(self, sample_result):
        sample_result.memory_required_gb = 150.0
        orchestrator = ProfilingOrchestrator()
        report = orchestrator.profile_simulation(sample_result)
        assert any("Memory utilization" in r for r in report.recommendations)


class TestTraceAnalyzer:
    def test_analyze_trace(self):
        trace_path = Path(__file__).parent / "fixtures" / "trace.jsonl"
        analyzer = TraceAnalyzer(str(trace_path))
        result = analyzer.analyze()
        assert result.feasible
        assert result.tokens_total > 0
        assert len(result.per_step) == 2
