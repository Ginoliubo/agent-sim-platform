"""Multi-agent swarm workload preset.

Models a parallel agent swarm where parent agent spawns sub-agents for
exploration, planning, or validation. Higher concurrency, shorter individual
steps, and more communication overhead.
"""

from .base import WorkloadSpec

MULTI_AGENT_WORKLOAD = WorkloadSpec(
    name="multi-agent",
    max_steps=30,
    avg_steps=12.0,
    step_std=6.0,
    context_limit=128000,
    token_distributions={
        "thought": (280.0, 80.0),
        "action": (90.0, 40.0),
        "observation": (500.0, 400.0),
    },
    tool_delays={
        "open": 0.04,
        "view": 0.02,
        "edit": 0.06,
        "bash_test": 6.0,
        "bash_install": 18.0,
        "search": 0.4,
        "submit": 0.01,
        "delegate": 0.5,
    },
    tool_probs={
        "open": 0.20,
        "view": 0.18,
        "bash_test": 0.12,
        "bash_install": 0.03,
        "edit": 0.15,
        "search": 0.10,
        "submit": 0.02,
        "delegate": 0.20,
    },
    env_init_delay=20.0,
    description="Parallel agent swarm with sub-agent delegation and shorter per-agent steps.",
)

__all__ = ["MULTI_AGENT_WORKLOAD"]
