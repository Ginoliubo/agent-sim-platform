"""Calibration engine: compare simulation outputs against benchmark fixtures."""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from ..algorithms import DEFAULT_REGISTRY as ALGORITHM_REGISTRY
from ..data_models import (
    AFDConfig,
    BenchmarkCase,
    CalibrationConfig,
    InferenceServiceConfig,
    OptimizationConfig,
    PDConfig,
    ParallelismConfig,
    TrainingConfig,
)
from ..hardware import DEFAULT_REGISTRY as HW_REGISTRY
from ..hardware import DEFAULT_CLUSTER_REGISTRY, ClusterSpec
from ..models import DEFAULT_REGISTRY as MODEL_REGISTRY
from ..simulation import run_serving, run_training
from ..simulation.capacity import CapacityEstimator
from .fitting import (
    ConstantFitter,
    apply_per_fixture_overrides,
    override_constants,
)
from .metrics import aggregate_metric_errors, compute_errors, mape, rmse
from .report import CalibrationReport


class CalibrationEngine:
    """Evaluate and fit simulation constants against benchmark fixtures."""

    # Map fixture domain to the metrics we compare
    DOMAIN_METRICS = {
        "training": ["total_time_seconds", "mfu", "memory_per_gpu_gb"],
        "serving": [
            "throughput_req_per_sec",
            "throughput_tok_per_sec",
            "ttft_p50_ms",
            "ttft_p99_ms",
            "tpot_p50_ms",
            "tpot_p99_ms",
        ],
        "capacity": ["memory_required_gb", "decode_latency_per_token_ms"],
    }

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()

    def _resolve_model(self, fixture: BenchmarkCase, overrides: Dict[str, float]) -> Tuple[object, OptimizationConfig]:
        """Resolve model and optimization with overrides applied."""
        model = MODEL_REGISTRY.get(fixture.model_name)
        algorithm = ALGORITHM_REGISTRY.get(fixture.algorithm_name)
        if model.algorithm_family.name != algorithm.name:
            model = replace(model, algorithm_family=algorithm)

        optimization = OptimizationConfig(name="calibration")
        # Apply fixture-level optimization overrides (e.g. empirical overheads)
        opt_overrides_from_fixture = {}
        for key in ("prefill_overhead_ms", "decode_overhead_ms"):
            if key in fixture.config:
                opt_overrides_from_fixture[key] = fixture.config[key]
        if opt_overrides_from_fixture:
            optimization = replace(optimization, **opt_overrides_from_fixture)
        model, optimization = apply_per_fixture_overrides(
            model, optimization, overrides
        )
        return model, optimization

    def _build_training_config(
        self, fixture: BenchmarkCase, overrides: Dict[str, float]
    ) -> TrainingConfig:
        """Build TrainingConfig from fixture, applying overrides."""
        cfg_dict = dict(fixture.config)
        parallelism = cfg_dict.pop("parallelism", {})
        cfg_dict["parallelism"] = ParallelismConfig(**parallelism)
        if "mfu_target" in overrides:
            cfg_dict["mfu_target"] = overrides["mfu_target"]
        return TrainingConfig(**cfg_dict)

    def _build_service_config(self, fixture: BenchmarkCase) -> InferenceServiceConfig:
        """Build InferenceServiceConfig from fixture, handling nested PD/AFD dicts."""
        cfg_dict = dict(fixture.config)
        # Remove keys consumed by parallelism / cluster / calibration resolution,
        # not by InferenceServiceConfig.
        for key in ("tp", "pp", "cp", "gpu_count", "cluster", "decode_overhead_ms", "prefill_overhead_ms"):
            cfg_dict.pop(key, None)
        pd_cfg = cfg_dict.pop("pd_config", None)
        if isinstance(pd_cfg, dict):
            cfg_dict["pd_config"] = PDConfig(**pd_cfg)
        afd_cfg = cfg_dict.pop("afd_config", None)
        if isinstance(afd_cfg, dict):
            cfg_dict["afd_config"] = AFDConfig(**afd_cfg)
        return InferenceServiceConfig(**cfg_dict)

    def _resolve_serving_kwargs(self, fixture: BenchmarkCase) -> Dict[str, object]:
        """Infer GPU count, cluster, and parallelism for a serving fixture."""
        cfg = dict(fixture.config)
        kwargs: Dict[str, object] = {}

        # Cluster
        cluster_name = cfg.get("cluster")
        if cluster_name:
            kwargs["cluster"] = DEFAULT_CLUSTER_REGISTRY.get(cluster_name)

        # Determine total GPU count from explicit config or PD/AFD pools
        gpu_count = cfg.get("gpu_count")
        pd_cfg = cfg.get("pd_config")
        afd_cfg = cfg.get("afd_config")
        pool_total = 0
        if gpu_count is None and isinstance(pd_cfg, dict) and pd_cfg.get("enabled"):
            pool_total = pd_cfg.get("prefill_gpu_count", 0) + pd_cfg.get("decode_gpu_count", 0)
        if gpu_count is None and isinstance(afd_cfg, dict) and afd_cfg.get("enabled"):
            pool_total = (
                afd_cfg.get("attention_gpu_count", 0)
                + afd_cfg.get("ffn_gpu_count", 0)
                + afd_cfg.get("decode_gpu_count", 0)
            )
        if gpu_count is None and pool_total > 0:
            gpu_count = pool_total
        if gpu_count:
            kwargs["gpu_count"] = int(gpu_count)

        # Parallelism: prefer explicit config.
        # For PD/AFD pools, infer a plausible (tp, pp, cp) split so that
        # tp*pp*cp == pool_total and memory/communication are shardable.
        model = MODEL_REGISTRY.get(fixture.model_name)
        tp = cfg.get("tp")
        pp = cfg.get("pp")
        cp = cfg.get("cp")
        if tp is None or pp is None or cp is None:
            if pool_total > 0:
                tp, pp, cp = self._infer_parallelism_for_pool(pool_total)
            else:
                tp = 8 if model.total_params_b >= 30 else 1
                pp = 1
                cp = 1
        kwargs.update({"tp": int(tp), "pp": int(pp), "cp": int(cp)})

        return kwargs

    @staticmethod
    def _infer_parallelism_for_pool(pool_total: int) -> Tuple[int, int, int]:
        """Infer tp/pp/cp for one disaggregated pool.

        Prefer a small per-request footprint (TP-only) so that multiple requests
        can run data-parallel inside the pool.  CP/PP are left at 1; the caller
        scales by running multiple instances of this (tp,pp,cp) group.
        """
        tp = min(8, max(1, pool_total))
        return tp, 1, 1

    def _extract_predicted_metrics(
        self, fixture: BenchmarkCase, result
    ) -> Dict[str, float]:
        """Extract predicted metrics from a SimulationResult."""
        metrics = {}
        if fixture.domain == "training":
            metrics["total_time_seconds"] = float(result.latency_seconds)
            metrics["mfu"] = float(result.metadata.get("mfu", result.utilization_gpu))
            metrics["memory_per_gpu_gb"] = float(
                result.metadata.get("memory_per_gpu_gb", 0.0)
            )
        elif fixture.domain == "serving":
            for key in [
                "throughput_req_per_sec",
                "throughput_tok_per_sec",
                "ttft_p50_ms",
                "ttft_p99_ms",
                "tpot_p50_ms",
                "tpot_p99_ms",
            ]:
                metrics[key] = float(result.metadata.get(key, 0.0))
        elif fixture.domain == "capacity":
            metrics["memory_required_gb"] = float(result.memory_required_gb)
            decode_s = result.metadata.get("decode_latency_per_token_s", 0.0)
            if decode_s == 0.0:
                decode_s = result.metadata.get("decode_latency_per_token_seconds", 0.0)
            metrics["decode_latency_per_token_ms"] = float(decode_s) * 1000.0
        return metrics

    def evaluate_fixture(
        self, fixture: BenchmarkCase, overrides: Optional[Dict[str, float]] = None
    ) -> Dict:
        """Run simulation for a single fixture and return observed/predicted/errors."""
        overrides = overrides or {}
        model, optimization = self._resolve_model(fixture, overrides)
        hardware = HW_REGISTRY.get(fixture.hardware_names[0])

        with override_constants(overrides):
            if fixture.domain == "training":
                training_config = self._build_training_config(fixture, overrides)
                result = run_training(model, hardware, training_config)
            elif fixture.domain == "serving":
                service_config = self._build_service_config(fixture)
                serving_kwargs = self._resolve_serving_kwargs(fixture)
                result = run_serving(
                    model,
                    hardware,
                    service_config,
                    optimization=optimization,
                    **serving_kwargs,
                )
            elif fixture.domain == "capacity":
                cfg = dict(fixture.config)
                context_tokens = cfg.pop("context_tokens")
                estimator = CapacityEstimator(model, hardware, **cfg)
                result = estimator.estimate(context_tokens)
            else:
                raise ValueError(f"Unsupported benchmark domain: {fixture.domain}")

        predicted = self._extract_predicted_metrics(fixture, result)
        errors = compute_errors(fixture.observed_metrics, predicted)
        return {
            "name": fixture.name,
            "domain": fixture.domain,
            "observed": fixture.observed_metrics,
            "predicted": predicted,
            "errors": errors,
        }

    def _fixtures_to_evaluate(
        self, registry, overrides: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, Dict[str, float], Dict[str, float]]]:
        """Return list of (name, observed, predicted) for fixtures matching domain filter."""
        overrides = overrides or {}
        results = []
        for fixture in registry.list(domain=self.config.domain if self.config.domain != "all" else None):
            evaluated = self.evaluate_fixture(fixture, overrides)
            results.append(
                (
                    evaluated["name"],
                    evaluated["observed"],
                    evaluated["predicted"],
                )
            )
        return results

    def evaluate_registry(
        self, registry, overrides: Optional[Dict[str, float]] = None
    ) -> CalibrationReport:
        """Evaluate all matching fixtures and return a CalibrationReport."""
        fixtures_predictions = self._fixtures_to_evaluate(registry, overrides)
        metric_summary = aggregate_metric_errors(fixtures_predictions)

        observed_all = []
        predicted_all = []
        for _, observed, predicted in fixtures_predictions:
            for key in observed:
                if key in predicted:
                    observed_all.append(float(observed[key]))
                    predicted_all.append(float(predicted[key]))

        overall_mape = mape(observed_all, predicted_all)
        overall_rmse = rmse(observed_all, predicted_all)

        fixture_results = []
        for name, observed, predicted in fixtures_predictions:
            fixture_results.append(
                {
                    "name": name,
                    "domain": self.config.domain,
                    "observed": observed,
                    "predicted": predicted,
                    "errors": compute_errors(observed, predicted),
                }
            )

        return CalibrationReport(
            fixture_results=fixture_results,
            metric_summary=metric_summary,
            overall_mape=overall_mape,
            overall_rmse=overall_rmse,
        )

    def fit(self, registry) -> CalibrationReport:
        """Run auto-fitting and return a CalibrationReport with fitted constants."""
        baseline_report = self.evaluate_registry(registry)

        def evaluate_fn(values: Dict[str, float]) -> CalibrationReport:
            return self.evaluate_registry(registry, overrides=values)

        fitter = ConstantFitter(self.config, evaluate_fn)
        best_values, history = fitter.fit()

        final_report = self.evaluate_registry(registry, overrides=best_values)
        final_report.fitted_values = {
            k: v for k, v in best_values.items() if k in self.config.fit_params
        }
        final_report.iterations = history

        # Preserve baseline comparison in metadata
        final_report.fixture_results.append(
            {
                "name": "__baseline_summary__",
                "domain": "meta",
                "observed": {"baseline_overall_mape": baseline_report.overall_mape},
                "predicted": {"fitted_overall_mape": final_report.overall_mape},
                "errors": {
                    "mape_improvement": {
                        "observed": baseline_report.overall_mape,
                        "predicted": final_report.overall_mape,
                        "abs": baseline_report.overall_mape - final_report.overall_mape,
                        "pct": (
                            (baseline_report.overall_mape - final_report.overall_mape)
                            / baseline_report.overall_mape
                            if baseline_report.overall_mape > 0
                            else 0.0
                        ),
                    }
                },
            }
        )

        return final_report


__all__ = ["CalibrationEngine"]
