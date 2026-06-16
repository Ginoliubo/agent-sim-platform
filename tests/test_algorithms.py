"""Tests for algorithm family framework."""

import pytest

from agent_sim_platform.algorithms import DEFAULT_REGISTRY, DENSE, MAMBA, MOE
from agent_sim_platform.models import DEFAULT_REGISTRY as MODEL_REGISTRY


def test_algorithm_registry():
    names = DEFAULT_REGISTRY.names()
    assert "dense" in names
    assert "moe" in names
    assert "mamba" in names
    assert "linear_attention" in names
    assert "ring_attention" in names


def test_dense_kv_cache():
    model = MODEL_REGISTRY.get("70B-Dense")
    kv = DENSE.kv_bytes_per_token(model, "FP8")
    assert kv > 0


def test_mamba_no_kv_cache():
    model = MODEL_REGISTRY.get("70B-Dense")
    kv = MAMBA.kv_bytes_per_token(model, "FP8")
    assert kv == 0.0


def test_moe_uses_active_params_for_flops():
    model = MODEL_REGISTRY.get("10T-MoE")
    forward_flops = MOE.flops_per_token_forward(model)
    # 10T-MoE active = 1000B, so forward = 2 * 1000B = 2e12
    assert forward_flops == 2e12


def test_model_spec_algorithm_family_binding():
    dense = MODEL_REGISTRY.get("70B-Dense")
    moe = MODEL_REGISTRY.get("10T-MoE")
    assert dense.algorithm_family.name == "dense"
    assert moe.algorithm_family.name == "moe"


def test_model_flops_per_token():
    model = MODEL_REGISTRY.get("70B-Dense")
    assert model.flops_per_token_forward() == 2 * 70e9
    assert model.flops_per_token_backward() == 4 * 70e9
    assert model.flops_per_token_training() == 6 * 70e9
