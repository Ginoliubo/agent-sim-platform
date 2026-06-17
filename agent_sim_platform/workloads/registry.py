"""Workload registry."""

from typing import Dict, List

from .base import WorkloadSpec
from .claude_code import CLAUDE_CODE_WORKLOAD
from .massive_refactor import MASSIVE_REFACTOR_WORKLOAD
from .multi_agent import MULTI_AGENT_WORKLOAD
from .swe_agent import SWE_AGENT_WORKLOAD

WORKLOAD_PRESETS = [SWE_AGENT_WORKLOAD, CLAUDE_CODE_WORKLOAD, MULTI_AGENT_WORKLOAD, MASSIVE_REFACTOR_WORKLOAD]


class WorkloadRegistry:
    """Central registry for workload presets."""

    def __init__(self, specs=None):
        self._specs: Dict[str, WorkloadSpec] = {}
        if specs:
            for spec in specs:
                self.register(spec)

    def register(self, spec: WorkloadSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Workload '{spec.name}' already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> WorkloadSpec:
        if name not in self._specs:
            raise KeyError(f"Workload '{name}' not found. Available: {list(self._specs.keys())}")
        return self._specs[name]

    def list(self) -> List[WorkloadSpec]:
        return sorted(self._specs.values(), key=lambda s: s.name)

    def names(self) -> List[str]:
        return sorted(self._specs.keys())


DEFAULT_REGISTRY = WorkloadRegistry(WORKLOAD_PRESETS)
