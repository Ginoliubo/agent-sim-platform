"""Auto-fitting of hard-coded simulation constants."""

from contextlib import contextmanager
from dataclasses import replace
from typing import Dict, Iterator, List, Optional, Tuple

from .. import config as sim_config
from ..algorithms.base import AlgorithmFamily
from ..data_models import (
    CalibrationConfig,
    ModelSpec,
    OptimizationConfig,
)


# Parameter name -> (low, high, step)
PARAMETER_RANGES: Dict[str, Tuple[float, float, float]] = {
    "mfu_target": (0.10, 0.70, 0.05),
    "default_prefill_utilization": (0.10, 0.95, 0.05),
    "default_decode_utilization": (0.10, 0.90, 0.05),
    "default_prefill_attention_hbm_passes": (0.0, 2000.0, 100.0),
    "default_prefill_saturation_tokens": (0.0, 500000.0, 5000.0),
    "default_prefill_latency_floor_ms": (0.0, 500.0, 10.0),
    "activation_overhead_factor": (0.5, 2.0, 0.1),
    "continuous_batching_efficiency": (0.5, 2.0, 0.1),
    "kv_compression_ratio": (0.1, 1.0, 0.1),
}


def _config_constant_name(param: str) -> str:
    """Map a calibration parameter name to its config module constant name."""
    return param.upper()


def is_config_constant(param: str) -> bool:
    """Return True if the parameter is a module-level config constant."""
    return hasattr(sim_config, _config_constant_name(param))


@contextmanager
def override_constants(overrides: Dict[str, float]) -> Iterator[None]:
    """Temporarily override module-level config constants."""
    original: Dict[str, float] = {}
    for name, value in overrides.items():
        if is_config_constant(name):
            const_name = _config_constant_name(name)
            original[const_name] = getattr(sim_config, const_name)
            setattr(sim_config, const_name, value)
    try:
        yield
    finally:
        for const_name, value in original.items():
            setattr(sim_config, const_name, value)


def _parameter_values(param: str, current: float) -> List[float]:
    """Generate candidate values around the current best."""
    low, high, step = PARAMETER_RANGES.get(param, (current * 0.5, current * 1.5, current * 0.1))
    values = []
    v = low
    while v <= high + 1e-9:
        values.append(round(v, 6))
        v += step
    return values


def apply_per_fixture_overrides(
    model: ModelSpec,
    optimization: OptimizationConfig,
    overrides: Dict[str, float],
) -> Tuple[ModelSpec, OptimizationConfig]:
    """Apply overrides that require creating new dataclass instances."""
    if "activation_overhead_factor" in overrides:
        family = model.algorithm_family or AlgorithmFamily(name="dense")
        new_family = replace(family, activation_overhead_factor=overrides["activation_overhead_factor"])
        model = replace(model, algorithm_family=new_family)

    opt_overrides = {}
    if "continuous_batching_efficiency" in overrides:
        opt_overrides["continuous_batching_efficiency"] = overrides["continuous_batching_efficiency"]
    if "kv_compression_ratio" in overrides:
        opt_overrides["kv_compression_ratio"] = overrides["kv_compression_ratio"]
    if opt_overrides:
        optimization = replace(optimization, **opt_overrides)

    return model, optimization


class ConstantFitter:
    """Coordinate-descent fitter for simulation constants."""

    def __init__(
        self,
        config: CalibrationConfig,
        evaluate_fn,
        initial_values: Optional[Dict[str, float]] = None,
    ):
        self.config = config
        self.evaluate_fn = evaluate_fn
        self.initial_values = initial_values or self._baseline_values()
        self.best_values = dict(self.initial_values)

    @staticmethod
    def _baseline_values() -> Dict[str, float]:
        """Return baseline values for all tunable parameters."""
        return {
            "mfu_target": 0.35,
            "default_prefill_utilization": sim_config.DEFAULT_PREFILL_UTILIZATION,
            "default_decode_utilization": sim_config.DEFAULT_DECODE_UTILIZATION,
            "default_prefill_attention_hbm_passes": sim_config.DEFAULT_PREFILL_ATTENTION_HBM_PASSES,
            "default_prefill_saturation_tokens": sim_config.DEFAULT_PREFILL_SATURATION_TOKENS,
            "default_prefill_latency_floor_ms": sim_config.DEFAULT_PREFILL_LATENCY_FLOOR_MS,
            "activation_overhead_factor": 1.0,
            "continuous_batching_efficiency": 1.0,
            "kv_compression_ratio": 1.0,
        }

    def _current_error(self, values: Dict[str, float]) -> float:
        """Evaluate the calibration error for a given set of constants."""
        report = self.evaluate_fn(values)
        return report.overall_mape

    def fit(self) -> Tuple[Dict[str, float], List[Dict]]:
        """Run coordinate descent and return best constants plus iteration log."""
        fit_params = [
            p for p in self.config.fit_params if p in PARAMETER_RANGES
        ]
        if not fit_params:
            return dict(self.best_values), []

        history = []
        best_error = self._current_error(self.best_values)
        history.append(
            {"iteration": 0, "values": dict(self.best_values), "mape": best_error}
        )

        for iteration in range(1, self.config.max_iterations + 1):
            improved = False

            for param in fit_params:
                current_value = self.best_values[param]
                for value in _parameter_values(param, current_value):
                    candidate = dict(self.best_values)
                    candidate[param] = value
                    error = self._current_error(candidate)
                    if error < best_error - self.config.tolerance:
                        best_error = error
                        self.best_values = candidate
                        improved = True

                # Re-evaluate current best to pick up any changes from other params
                best_error = self._current_error(self.best_values)

            history.append(
                {
                    "iteration": iteration,
                    "values": dict(self.best_values),
                    "mape": best_error,
                }
            )

            if not improved:
                break

        return dict(self.best_values), history


__all__ = [
    "ConstantFitter",
    "PARAMETER_RANGES",
    "apply_per_fixture_overrides",
    "override_constants",
]
