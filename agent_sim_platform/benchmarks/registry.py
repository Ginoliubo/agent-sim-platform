"""Benchmark registry."""

from typing import Dict, List, Optional

from ..data_models import BenchmarkCase


class BenchmarkRegistry:
    """Central registry for benchmark fixtures."""

    def __init__(self):
        self._fixtures: Dict[str, BenchmarkCase] = {}

    def register(self, fixture: BenchmarkCase) -> None:
        """Register a benchmark fixture."""
        if fixture.name in self._fixtures:
            raise ValueError(f"Benchmark fixture '{fixture.name}' already registered")
        self._fixtures[fixture.name] = fixture

    def get(self, name: str) -> BenchmarkCase:
        """Retrieve a fixture by name."""
        if name not in self._fixtures:
            raise KeyError(
                f"Benchmark fixture '{name}' not found. Available: {list(self._fixtures.keys())}"
            )
        return self._fixtures[name]

    def list(
        self,
        domain: Optional[str] = None,
    ) -> List[BenchmarkCase]:
        """List fixtures, optionally filtered by domain."""
        results = list(self._fixtures.values())
        if domain:
            results = [f for f in results if f.domain == domain]
        return sorted(results, key=lambda f: (f.domain, f.name))

    def names(self) -> List[str]:
        """Return all registered fixture names."""
        return sorted(self._fixtures.keys())

    def __len__(self) -> int:
        return len(self._fixtures)


__all__ = ["BenchmarkRegistry"]
