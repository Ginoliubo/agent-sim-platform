"""YAML/JSON benchmark fixture loader."""

import json
from pathlib import Path
from typing import List, Union

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to load benchmark fixtures. Install with: pip install pyyaml"
    ) from exc

from ..data_models import BenchmarkCase
from .registry import BenchmarkRegistry


def _load_file(path: Path) -> dict:
    """Load a YAML or JSON file into a dictionary."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    if path.suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported fixture format: {path.suffix}")


def _build_fixture(data: dict) -> BenchmarkCase:
    """Validate and build a BenchmarkCase from raw dict."""
    required = {"name", "domain", "source", "hardware_names", "model_name", "config", "observed_metrics"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Benchmark fixture missing required fields: {missing}")

    domain = data["domain"]
    if domain not in {"training", "serving", "capacity"}:
        raise ValueError(f"Unsupported benchmark domain: {domain}")

    return BenchmarkCase(
        name=data["name"],
        domain=domain,
        source=data["source"],
        source_url=data.get("source_url", ""),
        hardware_names=list(data["hardware_names"]),
        model_name=data["model_name"],
        algorithm_name=data.get("algorithm_name", "dense"),
        config=dict(data.get("config", {})),
        observed_metrics=dict(data.get("observed_metrics", {})),
        tolerance=dict(data.get("tolerance", {})),
        notes=data.get("notes", ""),
    )


def load_fixture(path: Union[str, Path]) -> BenchmarkCase:
    """Load a single fixture from a YAML/JSON file."""
    path = Path(path)
    data = _load_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Fixture file {path} must contain a mapping")
    return _build_fixture(data)


def load_fixtures_from_directory(directory: Union[str, Path]) -> List[BenchmarkCase]:
    """Load all YAML/JSON fixtures from a directory."""
    directory = Path(directory)
    fixtures: List[BenchmarkCase] = []
    for ext in ("*.yaml", "*.yml", "*.json"):
        for path in sorted(directory.glob(ext)):
            fixtures.append(load_fixture(path))
    return fixtures


def _build_default_registry() -> BenchmarkRegistry:
    """Build the default registry from bundled fixtures."""
    registry = BenchmarkRegistry()
    fixtures_dir = Path(__file__).with_name("fixtures")
    for fixture in load_fixtures_from_directory(fixtures_dir):
        registry.register(fixture)
    return registry


DEFAULT_REGISTRY = _build_default_registry()

__all__ = [
    "DEFAULT_REGISTRY",
    "load_fixture",
    "load_fixtures_from_directory",
]
