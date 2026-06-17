"""Workload presets and registry."""

from .base import WorkloadSpec
from .claude_code import CLAUDE_CODE_WORKLOAD
from .massive_refactor import MASSIVE_REFACTOR_WORKLOAD
from .multi_agent import MULTI_AGENT_WORKLOAD
from .registry import DEFAULT_REGISTRY, WORKLOAD_PRESETS, WorkloadRegistry
from .swe_agent import SWE_AGENT_WORKLOAD

__all__ = [
    "WorkloadSpec",
    "WorkloadRegistry",
    "DEFAULT_REGISTRY",
    "WORKLOAD_PRESETS",
    "SWE_AGENT_WORKLOAD",
    "CLAUDE_CODE_WORKLOAD",
    "MULTI_AGENT_WORKLOAD",
    "MASSIVE_REFACTOR_WORKLOAD",
]
