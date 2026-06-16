"""Tests for capacity estimator."""

import pytest

from agent_sim_platform.hardware import DEFAULT_REGISTRY as HW_REGISTRY
from agent_sim_platform.models import DEFAULT_REGISTRY as MODEL_REGISTRY
from agent_sim_platform.simulation import CapacityEstimator


def test_capacity_fits_small_config():
    est = CapacityEstimator(
        model=MODEL_REGISTRY.get("70B-Dense"),
        hardware=HW_REGISTRY.get("H100-SXM5"),
        precision="FP16",
        tp=8,
        pp=1,
    )
    assert est.fits(32768)


def test_capacity_overflow_large_context():
    est = CapacityEstimator(
        model=MODEL_REGISTRY.get("10T-MoE"),
        hardware=HW_REGISTRY.get("H100-SXM5"),
        precision="FP8",
        tp=8,
        pp=1,
    )
    assert not est.fits(10_000_000)


def test_capacity_future_hardware():
    est = CapacityEstimator(
        model=MODEL_REGISTRY.get("10T-MoE"),
        hardware=HW_REGISTRY.get("Feynman"),
        precision="FP4",
        tp=8,
        pp=8,
    )
    result = est.estimate(1_000_000)
    assert result.gpu_count == 64
    assert result.metadata["decode_tps"] > 0
