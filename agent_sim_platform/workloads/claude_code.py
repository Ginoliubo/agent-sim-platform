"""Claude Code / IDE assistant workload preset.

Models the multi-step workflow described in the coding-agent workflow deep dive:
input understanding, context assembly, planning, repo understanding, ReAct loop,
validation, error fix, self-review, and git/PR steps.
"""

from .base import WorkloadSpec

CLAUDE_CODE_WORKLOAD = WorkloadSpec(
    name="claude-code",
    max_steps=50,
    avg_steps=18.0,
    step_std=10.0,
    context_limit=200000,
    token_distributions={
        "thought": (500.0, 150.0),
        "action": (180.0, 80.0),
        "observation": (1200.0, 900.0),
    },
    tool_delays={
        "open": 0.05,
        "view": 0.03,
        "edit": 0.08,
        "bash_test": 12.0,
        "bash_install": 30.0,
        "search": 0.8,
        "submit": 0.01,
    },
    tool_probs={
        "open": 0.22,
        "view": 0.20,
        "bash_test": 0.18,
        "bash_install": 0.04,
        "edit": 0.20,
        "search": 0.12,
        "submit": 0.04,
    },
    env_init_delay=30.0,
    description="IDE-integrated coding assistant with larger context and more search/edit heavy steps.",
)

__all__ = ["CLAUDE_CODE_WORKLOAD"]
