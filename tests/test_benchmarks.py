"""Tests for benchmark fixtures and registry."""

import pytest

from agent_sim_platform.benchmarks import DEFAULT_REGISTRY, load_fixture
from agent_sim_platform.benchmarks.loader import load_fixtures_from_directory
from agent_sim_platform.benchmarks.registry import BenchmarkRegistry


def test_default_registry_loads_all_fixtures():
    """The default registry should load all bundled YAML fixtures."""
    names = DEFAULT_REGISTRY.names()
    assert len(names) >= 6
    assert "llama2_70b_pretrain" in names
    assert "llama3_70b_pretrain" in names
    assert "deepseek_v3_pretrain" in names
    assert "vllm_llama70b_serving" in names
    assert "mixtral_8x22b_serving" in names
    assert "capacity_llama70b_4k" in names


@pytest.mark.parametrize(
    "name,domain",
    [
        ("llama2_70b_pretrain", "training"),
        ("llama3_70b_pretrain", "training"),
        ("deepseek_v3_pretrain", "training"),
        ("vllm_llama70b_serving", "serving"),
        ("mixtral_8x22b_serving", "serving"),
        ("capacity_llama70b_4k", "capacity"),
    ],
)
def test_fixture_domain(name, domain):
    fixture = DEFAULT_REGISTRY.get(name)
    assert fixture.domain == domain
    assert fixture.name == name
    assert fixture.source
    assert fixture.hardware_names
    assert fixture.model_name
    assert fixture.config
    assert fixture.observed_metrics


def test_registry_filter_by_domain():
    training = DEFAULT_REGISTRY.list(domain="training")
    serving = DEFAULT_REGISTRY.list(domain="serving")
    capacity = DEFAULT_REGISTRY.list(domain="capacity")
    assert len(training) >= 3
    assert len(serving) >= 2
    assert len(capacity) >= 1
    assert all(f.domain == "training" for f in training)
    assert all(f.domain == "serving" for f in serving)
    assert all(f.domain == "capacity" for f in capacity)


def test_registry_duplicate_raises():
    registry = BenchmarkRegistry()
    registry.register(DEFAULT_REGISTRY.get("llama2_70b_pretrain"))
    with pytest.raises(ValueError):
        registry.register(DEFAULT_REGISTRY.get("llama2_70b_pretrain"))


def test_load_fixture(tmp_path):
    fixture_path = tmp_path / "custom.yaml"
    fixture_path.write_text(
        """
name: custom_test
domain: capacity
source: test
hardware_names:
  - A100-SXM4
model_name: 70B-Dense
config:
  context_tokens: 1024
  precision: FP8
observed_metrics:
  fits: true
tolerance:
  mape: 0.10
""",
        encoding="utf-8",
    )
    fixture = load_fixture(fixture_path)
    assert fixture.name == "custom_test"
    assert fixture.domain == "capacity"
    assert fixture.tolerance["mape"] == 0.10


def test_load_fixtures_from_directory(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "name: a\ndomain: capacity\nsource: test\nhardware_names: [A100-SXM4]\nmodel_name: 70B-Dense\nconfig: {}\nobserved_metrics: {}\n",
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text(
        '{"name": "b", "domain": "capacity", "source": "test", "hardware_names": ["A100-SXM4"], "model_name": "70B-Dense", "config": {}, "observed_metrics": {}}',
        encoding="utf-8",
    )
    fixtures = load_fixtures_from_directory(tmp_path)
    assert len(fixtures) == 2
    assert {f.name for f in fixtures} == {"a", "b"}


def test_fixture_missing_required_field(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\ndomain: training\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_fixture(bad)


def test_fixture_invalid_domain(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\ndomain: unknown\nsource: test\nhardware_names: [A100-SXM4]\nmodel_name: 70B-Dense\nconfig: {}\nobserved_metrics: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_fixture(bad)
