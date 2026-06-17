"""Massive-scale refactoring workload preset.

Models agentic workloads such as:
- Google I/O 2026 Antigravity 2.0 demo: 93 agents building an OS in 12 hours
- Anthropic Claude Code large-scale multi-file refactoring
- Long-horizon coding tasks with high concurrency and large context windows
"""

from .base import WorkloadSpec

MASSIVE_REFACTOR_WORKLOAD = WorkloadSpec(
    name="massive-refactor",
    max_steps=200,
    avg_steps=80.0,
    step_std=40.0,
    context_limit=10000000,
    token_distributions={
        "thought": (500.0, 150.0),
        "action": (200.0, 80.0),
        "observation": (1500.0, 1000.0),
    },
    tool_delays={
        "open": 0.05,
        "view": 0.03,
        "edit": 0.15,
        "bash_test": 15.0,
        "bash_install": 30.0,
        "search": 1.0,
        "submit": 0.01,
    },
    tool_probs={
        "open": 0.20,
        "view": 0.20,
        "bash_test": 0.20,
        "bash_install": 0.05,
        "edit": 0.25,
        "search": 0.08,
        "submit": 0.02,
    },
    env_init_delay=120.0,
    description="Large-scale refactoring with long context, many edit/test cycles, and high concurrency.",
)

__all__ = ["MASSIVE_REFACTOR_WORKLOAD"]
