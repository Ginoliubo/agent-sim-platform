"""Hybrid analytical + ML residual calibration model.

The residual model learns a per-metric linear correction on top of the
analytical simulator baseline.  It is trained on benchmark fixtures and
intended to capture second-order effects (scheduler overhead, kernel launch,
FlashAttention tile efficiency, system-specific optimizations) that are hard
to express in closed-form analytical formulas.

This follows the industry-recognized approach from:
"Predicting LLM Inference Latency: A Roofline-Driven ML Method" (NeurIPS 2024).
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np


class ResidualModel:
    """Linear residual correction: prediction = baseline + X @ coef + intercept."""

    FEATURE_NAMES = [
        "log_seq_len",
        "log_model_params",
        "log_gpu_count",
        "is_pd",
        "is_afd",
        "is_moe",
        "log_cp",
    ]

    def __init__(self, coefficients: Optional[Dict[str, List[float]]] = None):
        """Initialize with optional pre-trained coefficients.

        coefficients: {metric_name: {"coef": [...], "intercept": float}}
        """
        self.coefficients: Dict[str, Dict[str, List[float]]] = coefficients or {}

    def _extract_features(
        self,
        seq_len: float,
        model_params_b: float,
        gpu_count: int,
        is_pd: bool,
        is_afd: bool,
        is_moe: bool,
        cp: int,
    ) -> np.ndarray:
        return np.array(
            [
                np.log1p(seq_len),
                np.log1p(model_params_b),
                np.log1p(gpu_count),
                float(is_pd),
                float(is_afd),
                float(is_moe),
                np.log1p(cp),
            ],
            dtype=np.float64,
        )

    def fit(
        self,
        fixtures: List[Tuple[str, Dict[str, float], Dict[str, float], Dict[str, float]]],
    ) -> "ResidualModel":
        """Fit a residual model from fixture features and baseline predictions.

        Args:
            fixtures: list of (name, features_dict, observed_metrics, predicted_metrics)

        Returns:
            self
        """
        # Group by metric
        metric_residuals: Dict[str, List[Tuple[np.ndarray, float]]] = {}

        for name, features, observed, predicted in fixtures:
            x = self._extract_features(**features)
            for metric, obs_val in observed.items():
                # Only fit latency metrics; throughput is a derived quantity
                # driven by the event loop and should not be directly corrected.
                if metric not in predicted or "throughput" in metric:
                    continue
                pred_val = predicted[metric]
                residual = float(obs_val) - float(pred_val)
                metric_residuals.setdefault(metric, []).append((x, residual))

        self.coefficients = {}
        for metric, samples in metric_residuals.items():
            if len(samples) < 3:
                # Not enough data to fit; use mean residual as intercept.
                mean_residual = sum(r for _, r in samples) / len(samples)
                self.coefficients[metric] = {
                    "coef": [0.0] * len(self.FEATURE_NAMES),
                    "intercept": float(mean_residual),
                }
                continue

            X = np.vstack([x for x, _ in samples])
            y = np.array([r for _, r in samples])

            # Ridge regression with tiny regularization for numerical stability.
            reg = 1e-3
            XtX = X.T @ X + reg * np.eye(X.shape[1])
            Xty = X.T @ y
            coef = np.linalg.solve(XtX, Xty)
            intercept = 0.0

            self.coefficients[metric] = {
                "coef": [float(c) for c in coef],
                "intercept": float(intercept),
            }

        return self

    def predict_residual(
        self,
        metric: str,
        seq_len: float,
        model_params_b: float,
        gpu_count: int,
        is_pd: bool,
        is_afd: bool,
        is_moe: bool,
        cp: int,
    ) -> float:
        """Predict residual for a single metric."""
        if metric not in self.coefficients:
            return 0.0
        coefs = self.coefficients[metric]
        x = self._extract_features(
            seq_len, model_params_b, gpu_count, is_pd, is_afd, is_moe, cp
        )
        return float(x @ np.array(coefs["coef"]) + coefs["intercept"])

    def apply(
        self,
        fixture_name: str,
        features: Dict[str, float],
        predicted: Dict[str, float],
    ) -> Dict[str, float]:
        """Return predicted metrics with residual correction applied."""
        corrected = dict(predicted)
        for metric in list(corrected.keys()):
            residual = self.predict_residual(metric, **features)
            corrected[metric] = max(0.0, corrected[metric] + residual)
        return corrected

    def to_dict(self) -> Dict:
        return {"feature_names": self.FEATURE_NAMES, "coefficients": self.coefficients}

    @classmethod
    def from_dict(cls, data: Dict) -> "ResidualModel":
        return cls(coefficients=data.get("coefficients", {}))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ResidualModel":
        with open(path) as f:
            return cls.from_dict(json.load(f))
