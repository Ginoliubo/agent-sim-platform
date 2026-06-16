"""Simulation runner for single and batch execution."""

from typing import Dict, List

import numpy as np

from ..data_models import SimulationConfig, SimulationResult
from .engine import SimulationEngine


def run_single(config: SimulationConfig) -> SimulationResult:
    """Run a single simulation."""
    return SimulationEngine(config).run()


def run_batch(
    config: SimulationConfig,
    n_runs: int = 100,
    seed_offset: int = 0,
) -> List[SimulationResult]:
    """Run multiple independent simulations with different seeds."""
    results = []
    for i in range(n_runs):
        cfg = config
        if n_runs > 1:
            cfg = SimulationConfig(
                **{**config.__dict__, "random_seed": config.random_seed + seed_offset + i}
            )
        results.append(SimulationEngine(cfg).run())
    return results


def summarize_results(results: List[SimulationResult]) -> Dict:
    """Aggregate a list of simulation results."""
    def _stats(values):
        arr = np.array(values, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    return {
        "count": len(results),
        "latency_seconds": _stats([r.latency_seconds for r in results]),
        "wall_time_seconds": _stats([r.wall_time_seconds for r in results]),
        "tokens_total": _stats([r.tokens_total for r in results]),
        "peak_kv_gb": _stats([r.peak_kv_gb for r in results]),
        "memory_required_gb": _stats([r.memory_required_gb for r in results]),
        "cost_usd": _stats([r.cost_usd for r in results]),
        "gpu_count": int(np.median([r.gpu_count for r in results])),
        "feasible_fraction": sum(r.feasible for r in results) / len(results),
    }
