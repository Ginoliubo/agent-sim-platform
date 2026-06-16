"""Tests for hardware registry."""

import pytest

from agent_sim_platform.hardware import DEFAULT_REGISTRY


def test_registry_contains_current_and_future():
    names = DEFAULT_REGISTRY.names()
    assert "H100-SXM5" in names
    assert "B200" in names
    assert "Ascend-910B" in names
    assert "Rubin" in names
    assert "Feynman" in names
    assert "TPU-v5p" in names


def test_future_filter():
    future = DEFAULT_REGISTRY.list(future_only=True)
    names = [s.name for s in future]
    assert "Rubin" in names
    assert "Ascend-970" in names
    assert "H100-SXM5" not in names


def test_vendor_filter():
    nvidia = DEFAULT_REGISTRY.list(vendor="nvidia")
    assert all(s.vendor == "nvidia" for s in nvidia)
    huawei = DEFAULT_REGISTRY.list(vendor="huawei")
    assert all(s.vendor == "huawei" for s in huawei)


def test_get_missing_raises():
    with pytest.raises(KeyError):
        DEFAULT_REGISTRY.get("nonexistent-gpu")
