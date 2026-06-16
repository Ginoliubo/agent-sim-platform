"""Tests for simulation engine."""

import pytest

from agent_sim_platform.config import OPT_BASELINE, OPT_LAYER2
from agent_sim_platform.data_models import AgentHarnessSpec, SimulationConfig
from agent_sim_platform.hardware import DEFAULT_REGISTRY as HW_REGISTRY
from agent_sim_platform.models import DEFAULT_REGISTRY as MODEL_REGISTRY
from agent_sim_platform.simulation import run_batch, run_simulation, summarize_results
from agent_sim_platform.workloads import DEFAULT_REGISTRY as WORKLOAD_REGISTRY


def _make_config(hw_name, model_name, opt):
    return SimulationConfig(
        hardware=HW_REGISTRY.get(hw_name),
        model=MODEL_REGISTRY.get(model_name),
        workload=WORKLOAD_REGISTRY.get("swe-agent"),
        harness=AgentHarnessSpec(name="default"),
        target_context_tokens=32768,
        precision="FP8",
        kv_precision="FP8",
        optimization=opt,
        random_seed=42,
    )


def test_run_simulation_basic():
    config = _make_config("H100-SXM5", "1T-MoE", OPT_BASELINE)
    result = run_simulation(config)
    assert result.latency_seconds > 0
    assert result.tokens_total > 0
    assert result.gpu_count >= 1
    assert result.bottleneck != ""


def test_optimization_speedup():
    config_baseline = _make_config("H100-SXM5", "1T-MoE", OPT_BASELINE)
    config_layer2 = _make_config("H100-SXM5", "1T-MoE", OPT_LAYER2)
    base = run_simulation(config_baseline)
    opt = run_simulation(config_layer2)
    assert opt.latency_seconds < base.latency_seconds


def test_infeasible_large_model():
    config = _make_config("H100-SXM5", "10T-MoE", OPT_BASELINE)
    config.target_context_tokens = 10_000_000
    result = run_simulation(config)
    # 10T-MoE @ 10M requires an impractical number of GPUs even if memory fits
    assert result.gpu_count > 500 or not result.feasible


def test_batch_consistency():
    config = _make_config("H100-SXM5", "1T-MoE", OPT_LAYER2)
    results = run_batch(config, n_runs=10)
    assert len(results) == 10
    summary = summarize_results(results)
    assert summary["count"] == 10
    assert summary["feasible_fraction"] >= 0.0


def test_future_hardware_rubin():
    config = _make_config("Rubin", "10T-MoE", OPT_LAYER2)
    result = run_simulation(config)
    assert result.config.hardware.is_future
    assert result.gpu_count > 0


def test_npu_ascend():
    config = _make_config("Ascend-910B", "70B-Dense", OPT_BASELINE)
    result = run_simulation(config)
    assert result.config.hardware.vendor == "huawei"
    assert result.latency_seconds > 0
