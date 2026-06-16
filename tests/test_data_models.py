"""Tests for core data models."""

import pytest

from agent_sim_platform.data_models import (
    BYTES_PER_PARAM,
    HardwareSpec,
    ModelSpec,
    normalize_precision,
)


def test_bytes_per_param():
    assert BYTES_PER_PARAM["FP32"] == 4.0
    assert BYTES_PER_PARAM["FP16"] == 2.0
    assert BYTES_PER_PARAM["FP8"] == 1.0
    assert BYTES_PER_PARAM["INT4"] == 0.5


def test_normalize_precision():
    assert normalize_precision("fp8") == "FP8"
    assert normalize_precision("bf16") == "BF16"
    assert normalize_precision("INT4") == "INT4"


def test_hardware_spec_effective_flops():
    hw = HardwareSpec(
        name="Test-GPU",
        vendor="test",
        kind="gpu",
        memory_gb=80,
        memory_bw_tb_s=3.0,
        fp16_tflops=1000.0,
        fp8_tflops=2000.0,
        fp4_tflops=4000.0,
    )
    assert hw.effective_flops("FP16", utilization=1.0) == 1000e12
    assert hw.effective_flops("FP8", utilization=1.0) == 2000e12
    assert hw.effective_flops("FP4", utilization=1.0) == 4000e12


def test_model_weight_memory():
    model = ModelSpec(
        name="Tiny",
        total_params_b=1,
        active_params_b=1,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_head=32,
    )
    assert abs(model.weight_memory_gb("FP8") - 0.931) < 0.01
    assert abs(model.weight_memory_gb("FP16") - 1.863) < 0.01


def test_model_kv_bytes_per_token():
    model = ModelSpec(
        name="Tiny",
        total_params_b=1,
        active_params_b=1,
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_head=32,
    )
    # n_kv_heads = 1, kv_per_token = 2 * 2 * 1 * 32 = 128 bytes at FP8
    assert model.kv_bytes_per_token("FP8") == 128
    assert model.kv_bytes_per_token("FP16") == 256


def test_model_gpu_needed():
    model = ModelSpec(
        name="Medium",
        total_params_b=100,
        active_params_b=100,
        n_layers=20,
        d_model=4096,
        n_heads=32,
        d_head=128,
    )
    # FP8 weights = ~93 GB, need 2x 80GB GPUs with overhead
    assert model.gpu_needed(80, "FP8", overhead=1.15) == 2
