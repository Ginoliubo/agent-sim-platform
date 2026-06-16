"""Calibration error metrics."""

from typing import Dict, List, Tuple


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, handling zeros."""
    if denominator == 0:
        if numerator == 0:
            return 0.0
        return float("inf")
    return numerator / denominator


def mape(observed: List[float], predicted: List[float]) -> float:
    """Mean Absolute Percentage Error.

    Returns average of |observed - predicted| / |observed|.
    If any observed value is zero, that sample is skipped.
    """
    errors = []
    for o, p in zip(observed, predicted):
        if o == 0:
            continue
        errors.append(abs(o - p) / abs(o))
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def rmse(observed: List[float], predicted: List[float]) -> float:
    """Root Mean Squared Error."""
    if not observed:
        return 0.0
    squared = [(o - p) ** 2 for o, p in zip(observed, predicted)]
    return (sum(squared) / len(squared)) ** 0.5


def r2(observed: List[float], predicted: List[float]) -> float:
    """Coefficient of determination."""
    if not observed:
        return 0.0
    mean_o = sum(observed) / len(observed)
    ss_res = sum((o - p) ** 2 for o, p in zip(observed, predicted))
    ss_tot = sum((o - mean_o) ** 2 for o in observed)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def compute_errors(
    observed: Dict[str, float], predicted: Dict[str, float]
) -> Dict[str, float]:
    """Compute per-metric absolute error and percentage error.

    Returns a dict mapping metric name to {"abs": ..., "pct": ..., "observed": ..., "predicted": ...}.
    """
    errors = {}
    for key in observed:
        if key not in predicted:
            continue
        o = float(observed[key])
        p = float(predicted[key])
        pct = _safe_divide(abs(o - p), abs(o)) if o != 0 else 0.0
        errors[key] = {
            "abs": abs(o - p),
            "pct": pct,
            "observed": o,
            "predicted": p,
        }
    return errors


def aggregate_metric_errors(
    fixtures_and_predictions: List[Tuple[str, Dict[str, float], Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-metric MAPE and RMSE across multiple fixtures.

    Input: list of (fixture_name, observed_dict, predicted_dict).
    Output: dict mapping metric_name -> {"mape": ..., "rmse": ..., "samples": ...}.
    """
    metric_values: Dict[str, Tuple[List[float], List[float]]] = {}
    for _, observed, predicted in fixtures_and_predictions:
        for key in observed:
            if key not in predicted:
                continue
            if key not in metric_values:
                metric_values[key] = ([], [])
            metric_values[key][0].append(float(observed[key]))
            metric_values[key][1].append(float(predicted[key]))

    result = {}
    for key, (obs, pred) in metric_values.items():
        result[key] = {
            "mape": mape(obs, pred),
            "rmse": rmse(obs, pred),
            "samples": len(obs),
        }
    return result


__all__ = [
    "mape",
    "rmse",
    "r2",
    "compute_errors",
    "aggregate_metric_errors",
]
