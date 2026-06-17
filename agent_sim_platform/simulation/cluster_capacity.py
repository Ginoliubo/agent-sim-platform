"""Cluster-level capacity estimator for large models and long contexts.

Given a model, hardware, cluster topology, and target context length, searches
over TP/PP/CP combinations to find the smallest feasible distributed inference
configuration and reports network feasibility.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..config import FEASIBILITY_MAX_MEMORY_UTIL
from ..data_models import (
    AgentHarnessSpec,
    HardwareSpec,
    ModelSpec,
    OptimizationConfig,
    SimulationConfig,
    SimulationResult,
    WorkloadSpec,
)
from ..hardware import ClusterSpec
from ..simulation.inference_serving import InferenceServingEngine
from ..utils.units import gb_to_bytes


@dataclass
class ClusterCapacityResult:
    """Result from cluster-level capacity estimation."""

    feasible: bool
    tp: int
    pp: int
    cp: int
    gpu_count: int
    memory_per_gpu_gb: float
    prefill_latency_s: float
    decode_latency_per_token_s: float
    cp_comm_time_ms: float
    tp_comm_time_ms: float
    pp_comm_time_ms: float
    bottleneck: str


class ClusterCapacityEstimator:
    """Estimate distributed inference capacity on a cluster."""

    def __init__(
        self,
        model: ModelSpec,
        hardware: HardwareSpec,
        cluster: ClusterSpec,
        precision: str = "FP8",
        kv_precision: str = "FP8",
        optimization: OptimizationConfig = None,
    ):
        self.model = model
        self.hardware = hardware
        self.cluster = cluster
        self.precision = precision.upper()
        self.kv_precision = kv_precision.upper()
        self.optimization = optimization or OptimizationConfig()

    def _fits(
        self,
        context_tokens: int,
        tp: int,
        pp: int,
        cp: int,
        batch_size: int = 1,
    ) -> Tuple[bool, float]:
        """Check whether a TP/PP/CP config fits per-GPU memory."""
        opt = self.optimization
        kv_bytes_per_token = (
            self.model.kv_bytes_per_token(self.kv_precision) * opt.kv_compression_ratio
        )
        bytes_per_param = self.model.bytes_per_param(self.precision)

        weight_per_gpu = self.model.weight_memory_gb(self.precision) / max(1, tp * pp)
        kv_per_gpu = (
            context_tokens * batch_size * kv_bytes_per_token / gb_to_bytes(1.0)
        ) / max(1, cp)
        activation_per_gpu = (
            2 * context_tokens * batch_size * self.model.d_model * bytes_per_param
        ) / max(1, tp * cp) / gb_to_bytes(1.0)
        activation_per_gpu += 2.0 * batch_size

        memory_per_gpu = weight_per_gpu + kv_per_gpu + activation_per_gpu
        fits = memory_per_gpu <= self.hardware.memory_gb * FEASIBILITY_MAX_MEMORY_UTIL
        return fits, memory_per_gpu

    def find_minimal_config(
        self,
        context_tokens: int,
        tp_candidates: List[int] = None,
        pp_candidates: List[int] = None,
        cp_candidates: List[int] = None,
        batch_size: int = 1,
    ) -> ClusterCapacityResult:
        """Search for the smallest feasible TP/PP/CP configuration."""
        tp_candidates = tp_candidates or [1, 2, 4, 8]
        pp_candidates = pp_candidates or [1, 2, 4, 8, 16, 32]
        cp_candidates = cp_candidates or [1, 2, 4, 8, 16, 32, 64, 128, 256]

        best: Optional[ClusterCapacityResult] = None

        for tp in tp_candidates:
            for pp in pp_candidates:
                for cp in cp_candidates:
                    total_gpus = tp * pp * cp
                    if total_gpus > self.cluster.total_gpus:
                        continue
                    fits, memory_per_gpu = self._fits(
                        context_tokens, tp, pp, cp, batch_size
                    )
                    if not fits:
                        continue
                    # Score: prefer fewer GPUs, then lower cp (less comm)
                    score = (total_gpus, cp, pp, tp)
                    if best is None or score < (
                        best.gpu_count,
                        best.cp,
                        best.pp,
                        best.tp,
                    ):
                        from ..data_models import InferenceServiceConfig

                        service_config = InferenceServiceConfig(
                            arrival_rate_per_sec=1.0,
                            request_length_mean=context_tokens,
                            output_length_mean=1,
                            simulation_duration_seconds=1.0,
                            max_batch_size=batch_size,
                        )
                        engine = InferenceServingEngine(
                            model=self.model,
                            hardware=self.hardware,
                            service_config=service_config,
                            precision=self.precision,
                            kv_precision=self.kv_precision,
                            optimization=self.optimization,
                            cluster=self.cluster,
                            tp=tp,
                            pp=pp,
                            cp=cp,
                        )
                        best = ClusterCapacityResult(
                            feasible=True,
                            tp=tp,
                            pp=pp,
                            cp=cp,
                            gpu_count=total_gpus,
                            memory_per_gpu_gb=memory_per_gpu,
                            prefill_latency_s=engine.prefill_time_per_token
                            * context_tokens,
                            decode_latency_per_token_s=engine.decode_time_per_token,
                            cp_comm_time_ms=engine._communication_breakdown(1, 1).cp_time_per_token_ms,
                            tp_comm_time_ms=engine._communication_breakdown(1, 1).tp_time_per_token_ms,
                            pp_comm_time_ms=engine._communication_breakdown(1, 1).pp_time_per_token_ms,
                            bottleneck="",
                        )

        if best is None:
            # Return the smallest infeasible config we tried for diagnostics
            return ClusterCapacityResult(
                feasible=False,
                tp=8,
                pp=4,
                cp=64,
                gpu_count=8 * 4 * 64,
                memory_per_gpu_gb=float("inf"),
                prefill_latency_s=float("inf"),
                decode_latency_per_token_s=float("inf"),
                cp_comm_time_ms=0.0,
                tp_comm_time_ms=0.0,
                pp_comm_time_ms=0.0,
                bottleneck="memory: HBM overflow on all tried configs",
            )
        return best

    def to_simulation_result(
        self, context_tokens: int, batch_size: int = 1
    ) -> SimulationResult:
        """Return a SimulationResult wrapping the minimal cluster config."""
        cap = self.find_minimal_config(context_tokens, batch_size=batch_size)

        sim_config = SimulationConfig(
            hardware=self.hardware,
            model=self.model,
            workload=WorkloadSpec(
                name="cluster-capacity",
                max_steps=1,
                avg_steps=1.0,
                step_std=0.0,
                context_limit=context_tokens,
            ),
            harness=AgentHarnessSpec(name="cluster-capacity", concurrency=1),
            target_context_tokens=context_tokens,
            precision=self.precision,
            kv_precision=self.kv_precision,
            tp=cap.tp,
            pp=cap.pp,
            batch_size=batch_size,
            optimization=self.optimization,
        )

        return SimulationResult(
            config=sim_config,
            latency_seconds=cap.prefill_latency_s,
            wall_time_seconds=cap.prefill_latency_s,
            tokens_total=context_tokens,
            tokens_input=context_tokens,
            tokens_output=0,
            peak_kv_gb=0.0,
            memory_required_gb=cap.memory_per_gpu_gb * cap.gpu_count,
            gpu_count=cap.gpu_count,
            feasible=cap.feasible,
            bottleneck=cap.bottleneck,
            cost_usd=0.0,
            utilization_gpu=0.0,
            metadata={
                "tp": cap.tp,
                "pp": cap.pp,
                "cp": cap.cp,
                "memory_per_gpu_gb": cap.memory_per_gpu_gb,
                "prefill_latency_s": cap.prefill_latency_s,
                "decode_latency_per_token_s": cap.decode_latency_per_token_s,
                "tp_comm_time_ms": cap.tp_comm_time_ms,
                "pp_comm_time_ms": cap.pp_comm_time_ms,
                "cp_comm_time_ms": cap.cp_comm_time_ms,
                "cluster": self.cluster.name,
                "topology": self.cluster.topology.name,
            },
        )


def estimate_cluster_capacity(
    model: ModelSpec,
    hardware: HardwareSpec,
    cluster: ClusterSpec,
    context_tokens: int,
    precision: str = "FP8",
    kv_precision: str = "FP8",
    optimization: OptimizationConfig = None,
    batch_size: int = 1,
) -> ClusterCapacityResult:
    """Convenience function for cluster capacity estimation."""
    estimator = ClusterCapacityEstimator(
        model=model,
        hardware=hardware,
        cluster=cluster,
        precision=precision,
        kv_precision=kv_precision,
        optimization=optimization,
    )
    return estimator.find_minimal_config(context_tokens, batch_size=batch_size)
