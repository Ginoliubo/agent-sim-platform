"""Tests for inference serving engine."""

from agent_sim_platform.data_models import InferenceServiceConfig
from agent_sim_platform.hardware import DEFAULT_REGISTRY as HW_REGISTRY
from agent_sim_platform.models import DEFAULT_REGISTRY as MODEL_REGISTRY
from agent_sim_platform.simulation.inference_serving import InferenceServingEngine, run_serving


def test_serving_basic():
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=5.0,
        request_length_mean=1024,
        output_length_mean=128,
        simulation_duration_seconds=30.0,
        max_batch_size=8,
    )
    result = run_serving(MODEL_REGISTRY.get("70B-Dense"), HW_REGISTRY.get("H100-SXM5"), cfg, gpu_count=8)
    assert result.metadata["requests_total"] > 0
    assert result.metadata["requests_completed"] >= 0
    assert result.metadata["throughput_req_per_sec"] >= 0


def test_serving_throughput_scaling():
    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("H100-SXM5")
    cfg_low = InferenceServiceConfig(
        arrival_rate_per_sec=1.0,
        request_length_mean=1024,
        output_length_mean=128,
        simulation_duration_seconds=20.0,
        max_batch_size=8,
    )
    cfg_high = InferenceServiceConfig(
        arrival_rate_per_sec=20.0,
        request_length_mean=1024,
        output_length_mean=128,
        simulation_duration_seconds=20.0,
        max_batch_size=8,
    )
    low = run_serving(model, hw, cfg_low, gpu_count=8)
    high = run_serving(model, hw, cfg_high, gpu_count=8)
    # Higher arrival rate should result in more drops or no more completed
    assert high.metadata["requests_dropped"] >= low.metadata["requests_dropped"]


def test_serving_tpot_positive():
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=2.0,
        request_length_mean=1024,
        output_length_mean=128,
        simulation_duration_seconds=30.0,
        max_batch_size=8,
    )
    result = run_serving(MODEL_REGISTRY.get("70B-Dense"), HW_REGISTRY.get("H100-SXM5"), cfg, gpu_count=8)
    if result.metadata["requests_completed"] > 0:
        assert result.metadata["tpot_p50_ms"] > 0


def test_serving_memory_does_not_fit():
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=10.0,
        request_length_mean=4096,
        output_length_mean=512,
        simulation_duration_seconds=10.0,
        max_batch_size=8,
    )
    result = run_serving(MODEL_REGISTRY.get("10T-MoE"), HW_REGISTRY.get("H100-SXM5"), cfg, gpu_count=8)
    # 10T-MoE does not fit on 8 H100s, so nothing should complete
    assert result.metadata["requests_completed"] == 0
