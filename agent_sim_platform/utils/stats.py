"""Statistical helpers for workload simulation."""

from typing import Dict, List, Tuple

import numpy as np


def clip_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    size: int,
    lower: float = 0.0,
) -> np.ndarray:
    """Sample from a normal distribution and clip below a lower bound."""
    samples = rng.normal(loc=mean, scale=std, size=size)
    return np.maximum(samples, lower)


def sample_discrete(rng: np.random.Generator, probs: Dict[str, float]) -> str:
    """Sample a key from a probability dictionary."""
    keys = list(probs.keys())
    values = np.array([probs[k] for k in keys], dtype=float)
    values /= values.sum()
    return rng.choice(keys, p=values)


def percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile of a list of floats."""
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def summary(values: List[float]) -> Dict[str, float]:
    """Return mean, std, p50, p90, p99 summary of a list."""
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


def token_distributions_to_array(
    rng: np.random.Generator,
    distributions: Dict[str, Tuple[float, float]],
    n: int,
) -> Dict[str, np.ndarray]:
    """Sample token counts for each distribution key."""
    return {
        key: clip_normal(rng, mean=mean, std=std, size=n)
        for key, (mean, std) in distributions.items()
    }
