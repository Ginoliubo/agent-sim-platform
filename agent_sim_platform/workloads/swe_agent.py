"""SWE-agent ReAct loop workload preset."""

from .base import WorkloadSpec

SWE_AGENT_WORKLOAD = WorkloadSpec(
    name="swe-agent",
    max_steps=100,
    avg_steps=25.0,
    step_std=15.0,
    context_limit=128000,
    token_distributions={
        "thought": (350.0, 100.0),
        "action": (120.0, 50.0),
        "observation": (800.0, 600.0),
    },
    tool_delays={
        "open": 0.05,
        "view": 0.03,
        "edit": 0.10,
        "bash_test": 8.0,
        "bash_install": 25.0,
        "search": 0.5,
        "submit": 0.01,
    },
    tool_probs={
        "open": 0.30,
        "view": 0.22,
        "bash_test": 0.15,
        "bash_install": 0.05,
        "edit": 0.16,
        "search": 0.08,
        "submit": 0.04,
    },
    env_init_delay=60.0,
    description="SWE-bench style ReAct agent with open/view/edit/bash_test/search tools.",
)

__all__ = ["SWE_AGENT_WORKLOAD"]
