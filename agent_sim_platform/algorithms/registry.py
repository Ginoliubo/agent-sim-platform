"""Algorithm family registry."""

from typing import Dict, List

from .base import AlgorithmFamily
from .families import ALGORITHM_FAMILIES


class AlgorithmRegistry:
    """Central registry for algorithm family presets."""

    def __init__(self, families=None):
        self._families: Dict[str, AlgorithmFamily] = {}
        if families:
            for fam in families:
                self.register(fam)

    def register(self, family: AlgorithmFamily) -> None:
        if family.name in self._families:
            raise ValueError(f"Algorithm family '{family.name}' already registered")
        self._families[family.name] = family

    def get(self, name: str) -> AlgorithmFamily:
        if name not in self._families:
            raise KeyError(
                f"Algorithm family '{name}' not found. Available: {list(self._families.keys())}"
            )
        return self._families[name]

    def list(self) -> List[AlgorithmFamily]:
        return sorted(self._families.values(), key=lambda f: f.name)

    def names(self) -> List[str]:
        return sorted(self._families.keys())


DEFAULT_REGISTRY = AlgorithmRegistry(ALGORITHM_FAMILIES)

__all__ = ["AlgorithmRegistry", "DEFAULT_REGISTRY"]
