"""Model registry."""

from typing import Dict, List, Optional

from .base import ModelSpec
from .presets import MODEL_PRESETS


class ModelRegistry:
    """Central registry for model presets."""

    def __init__(self, specs=None):
        self._specs: Dict[str, ModelSpec] = {}
        if specs:
            for spec in specs:
                self.register(spec)

    def register(self, spec: ModelSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Model '{spec.name}' already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ModelSpec:
        if name not in self._specs:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._specs.keys())}")
        return self._specs[name]

    def list(self, architecture: Optional[str] = None) -> List[ModelSpec]:
        results = []
        for spec in self._specs.values():
            if architecture and spec.architecture.lower() != architecture.lower():
                continue
            results.append(spec)
        return sorted(results, key=lambda s: s.total_params_b)

    def names(self) -> List[str]:
        return sorted(self._specs.keys())


DEFAULT_REGISTRY = ModelRegistry(MODEL_PRESETS)
