"""Training simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..config import FEASIBILITY_MAX_MEMORY_UTIL
from ..data_models import (
    AgentHarnessSpec,
    HardwareSpec,
    ModelSpec,
    OptimizationConfig,
    ParallelismConfig,
    SimulationConfig,
    SimulationResult,
    TrainingConfig,
    WorkloadSpec,
)
from ..hardware import ClusterSpec
from ..utils.units import gb_to_bytes


@dataclass
class TrainingResult:
    """Detailed result from a training simulation."""

    total_time_seconds: float
    total_flops: float
    step_time_seconds: float
    compute_time_seconds: float
    communication_time_seconds: float
    memory_per_gpu_gb: float
    gpu_count: int
    mfu: float
    bottleneck: str
    cost_usd: float


class TrainingEngine:
    """Simulate a distributed training job.

    Models compute, communication, and memory for transformer-based training.
    Supports topology-aware communication estimates when a ClusterSpec is
    provided.
    """

    def __init__(
        self,
        model: ModelSpec,
        hardware: HardwareSpec,
        training_config: TrainingConfig,
        precision: str = "FP8",
        cluster: ClusterSpec = None,
    ):
        self.model = model
        self.hardware = hardware
        self.training_config = training_config
        self.precision = precision.upper()
        self.cluster = cluster
        self.bytes_per_param = model.bytes_per_param(self.precision)

    def _gpus_per_node(self) -> int:
        if self.cluster is None:
            return self._estimate_gpu_count()
        return self.cluster.gpus_per_node

    def _node_count(self) -> int:
        if self.cluster is None:
            return 1
        return self.cluster.node_count

    def _intra_node_bw_bytes_s(self) -> float:
        """Aggregate intra-node bandwidth in bytes/s."""
        if self.cluster is None:
            return self.hardware.interconnect_bw_gb_s * 1e9
        return self.cluster.topology.intra_node_bw_gb_s * 1e9

    def _inter_node_bw_bytes_s(self) -> float:
        """Effective per-node inter-node bandwidth in bytes/s."""
        if self.cluster is None:
            # Fallback to hardware interconnect if no cluster topology
            bw = self.hardware.interconnect_bw_gb_s
            if bw <= 0:
                bw = self.hardware.pcie_bw_gb_s
            return bw * 1e9
        return self.cluster.topology.effective_inter_node_bw_gb_s * 1e9

    def _cross_node_fraction(self, group_size: int) -> float:
        """Fraction of traffic that crosses node boundaries for a parallel group."""
        gpus_per_node = self._gpus_per_node()
        if group_size <= gpus_per_node:
            return 0.0
        return max(0.0, (group_size - gpus_per_node) / group_size)

    def _communication_time(
        self,
        bytes_moved: float,
        group_size: int,
    ) -> float:
        """Estimate communication time splitting intra- and inter-node traffic."""
        if bytes_moved <= 0:
            return 0.0
        cross_frac = self._cross_node_fraction(group_size)
        intra_frac = 1.0 - cross_frac
        intra_bw = self._intra_node_bw_bytes_s()
        inter_bw = self._inter_node_bw_bytes_s()
        time = 0.0
        if intra_bw > 0:
            time += intra_frac * bytes_moved / intra_bw
        if cross_frac > 0 and inter_bw > 0:
            time += cross_frac * bytes_moved / inter_bw
        return time

    def _memory_per_gpu_gb(self, gpu_count: int) -> float:
        """Estimate memory required per GPU in GB."""
        cfg = self.training_config
        parallelism = cfg.parallelism
        p_total = self.model.total_params_b * 1e9
        p_active = self.model.active_params_b * 1e9
        seq_len = cfg.sequence_length
        micro_batch = cfg.micro_batch_size
        n_layers = self.model.n_layers
        d_model = self.model.d_model

        # Weight memory (sharded by ZeRO / TP / PP / EP)
        weight_bytes = p_total * self.bytes_per_param
        weight_sharding = self._weight_sharding_factor(parallelism)
        weight_per_gpu = weight_bytes / weight_sharding

        # Gradient memory
        grad_bytes = p_total * self.bytes_per_param
        grad_sharding = self._gradient_sharding_factor(parallelism)
        grad_per_gpu = grad_bytes / grad_sharding

        # Optimizer state memory
        opt_bytes_per_param = 8.0 if cfg.optimizer == "adamw" else 4.0
        optimizer_bytes = p_total * opt_bytes_per_param
        opt_sharding = self._optimizer_sharding_factor(parallelism)
        optimizer_per_gpu = optimizer_bytes / opt_sharding

        # Activation memory
        # Simplified: 2 * seq_len * micro_batch * d_model * n_layers * bytes
        activation_bytes = (
            2 * seq_len * micro_batch * d_model * n_layers * self.bytes_per_param
        )
        if cfg.gradient_checkpointing:
            activation_bytes *= 0.4  # ~60% savings

        # KV cache during training (full sequence, no compression for training)
        kv_bytes_per_token = self.model.kv_bytes_per_token(self.precision)
        kv_bytes = seq_len * micro_batch * kv_bytes_per_token

        total_bytes = weight_per_gpu + grad_per_gpu + optimizer_per_gpu + activation_bytes + kv_bytes
        return total_bytes / gb_to_bytes(1.0)

    def _weight_sharding_factor(self, parallelism: ParallelismConfig) -> float:
        # Weights are sharded across TP, PP, EP, ZeRO-3 DP
        factor = parallelism.tp * parallelism.pp * parallelism.ep
        if self.training_config.zero_stage == 3:
            factor *= parallelism.dp
        return max(1, factor)

    def _gradient_sharding_factor(self, parallelism: ParallelismConfig) -> float:
        factor = parallelism.tp * parallelism.pp * parallelism.ep
        if self.training_config.zero_stage >= 2:
            factor *= parallelism.dp
        return max(1, factor)

    def _optimizer_sharding_factor(self, parallelism: ParallelismConfig) -> float:
        factor = parallelism.tp * parallelism.pp * parallelism.ep
        if self.training_config.zero_stage >= 1:
            factor *= parallelism.dp
        return max(1, factor)

    def _compute_time_per_step(self, gpu_count: int) -> float:
        """Compute time for one global step."""
        cfg = self.training_config
        tokens_per_step = cfg.global_batch_size * cfg.sequence_length
        flops_per_token = self.model.flops_per_token_training()
        total_flops = flops_per_token * tokens_per_step

        effective_flops = (
            self.hardware.effective_flops(self.precision, cfg.mfu_target) * gpu_count
        )
        return total_flops / effective_flops

    def _communication_breakdown_per_step(
        self, gpu_count: int, compute_time: float
    ) -> Dict[str, float]:
        """Breakdown of communication time per step by parallelism type (seconds)."""
        cfg = self.training_config
        parallelism = cfg.parallelism
        model_size_bytes = self.model.total_params_b * 1e9 * self.bytes_per_param

        breakdown: Dict[str, float] = {}

        # Data parallel gradient All-Reduce
        if parallelism.dp > 1 and cfg.zero_stage < 3:
            bytes_moved = 2 * (parallelism.dp - 1) / parallelism.dp * model_size_bytes
            breakdown["dp_allreduce"] = self._communication_time(bytes_moved, parallelism.dp)

        # Tensor parallel all-reduce per layer
        if parallelism.tp > 1:
            layer_size_bytes = model_size_bytes / self.model.n_layers
            bytes_per_layer = 2 * layer_size_bytes  # two all-reduces per layer
            bytes_moved = self.model.n_layers * bytes_per_layer
            breakdown["tp_allreduce"] = self._communication_time(bytes_moved, parallelism.tp)

        # Pipeline parallel bubble
        if parallelism.pp > 1:
            num_micro_batches = max(
                parallelism.pp * 4,
                cfg.global_batch_size // (cfg.micro_batch_size * parallelism.dp),
            )
            bubble_ratio = (parallelism.pp - 1) / (num_micro_batches + parallelism.pp - 1)
            breakdown["pp_bubble"] = compute_time * bubble_ratio

        # MoE expert parallel all-to-all
        if self.model.is_moe and parallelism.ep > 1:
            hidden_bytes = (
                cfg.global_batch_size
                * cfg.sequence_length
                * self.model.d_model
                * self.bytes_per_param
            )
            bytes_moved = 2 * (parallelism.ep - 1) / parallelism.ep * hidden_bytes
            breakdown["ep_alltoall"] = self._communication_time(bytes_moved, parallelism.ep)

        return breakdown

    def _communication_time_per_step(
        self, gpu_count: int, compute_time: float
    ) -> float:
        """Estimate communication time per step using topology-aware bandwidth."""
        return sum(self._communication_breakdown_per_step(gpu_count, compute_time).values())

    def _network_utilization(
        self,
        communication_breakdown: Dict[str, float],
        step_time: float,
        gpu_count: int,
    ) -> float:
        """Estimate peak inter-node network bandwidth utilization (0-1).

        Uses the largest single communication payload and step time to bound
        peak utilization; sustained utilization will be lower due to overlap.
        """
        if step_time <= 0 or not communication_breakdown:
            return 0.0

        cfg = self.training_config
        parallelism = cfg.parallelism
        model_size_bytes = self.model.total_params_b * 1e9 * self.bytes_per_param

        # Largest payload: DP all-reduce or EP all-to-all
        max_bytes = 0.0
        if parallelism.dp > 1 and cfg.zero_stage < 3:
            max_bytes = max(
                max_bytes,
                2 * (parallelism.dp - 1) / parallelism.dp * model_size_bytes,
            )
        if self.model.is_moe and parallelism.ep > 1:
            hidden_bytes = (
                cfg.global_batch_size
                * cfg.sequence_length
                * self.model.d_model
                * self.bytes_per_param
            )
            max_bytes = max(
                max_bytes,
                2 * (parallelism.ep - 1) / parallelism.ep * hidden_bytes,
            )

        bytes_per_sec = max_bytes / step_time

        if self.cluster is not None:
            total_inter_node_bw = (
                self.cluster.topology.aggregate_inter_node_bw_gb_s
                * self.cluster.node_count
                * 1e9
            )
        else:
            total_inter_node_bw = self.hardware.interconnect_bw_gb_s * gpu_count * 1e9

        if total_inter_node_bw <= 0:
            return 0.0
        return min(1.0, bytes_per_sec / total_inter_node_bw)

    def _estimate_gpu_count(self) -> int:
        """Estimate required GPU count based on memory."""
        cfg = self.training_config
        requested = cfg.parallelism.total_gpus
        if requested > 1:
            # Validate memory fits with requested parallelism
            memory_per_gpu = self._memory_per_gpu_gb(requested)
            if memory_per_gpu <= self.hardware.memory_gb * FEASIBILITY_MAX_MEMORY_UTIL:
                return requested

        # Otherwise scale up until it fits
        gpu_count = requested
        while gpu_count < 65536:
            memory_per_gpu = self._memory_per_gpu_gb(gpu_count)
            if memory_per_gpu <= self.hardware.memory_gb * FEASIBILITY_MAX_MEMORY_UTIL:
                return gpu_count
            gpu_count *= 2
        return gpu_count

    def run(self) -> SimulationResult:
        """Run training simulation and return unified result."""
        cfg = self.training_config
        gpu_count = self._estimate_gpu_count()
        memory_per_gpu = self._memory_per_gpu_gb(gpu_count)

        compute_time = self._compute_time_per_step(gpu_count)
        comm_breakdown = self._communication_breakdown_per_step(gpu_count, compute_time)
        communication_time = sum(comm_breakdown.values())
        data_loading_time = compute_time * cfg.data_loading_overhead_fraction

        step_time = compute_time + communication_time + data_loading_time
        total_time = step_time * cfg.steps
        total_flops = self.model.flops_per_token_training() * cfg.total_tokens

        # Actual MFU if we hit the target
        mfu = cfg.mfu_target

        # Bottleneck
        if memory_per_gpu > self.hardware.memory_gb:
            bottleneck = "memory: HBM overflow"
        elif communication_time > compute_time * 0.5:
            # Distinguish intra-node vs cross-node communication bottleneck
            cross_node_frac = 0.0
            if self.cluster is not None:
                for group_size in (cfg.parallelism.dp, cfg.parallelism.tp, cfg.parallelism.ep):
                    if group_size > 1:
                        cross_node_frac = max(cross_node_frac, self._cross_node_fraction(group_size))
            if cross_node_frac > 0.5:
                bottleneck = "network: cross-node interconnect bottleneck"
            else:
                bottleneck = "network: intra-node interconnect bottleneck"
        elif data_loading_time > compute_time * 0.2:
            bottleneck = "data-loading: IO bottleneck"
        else:
            bottleneck = "compute: GPU-bound"

        cost_usd = (total_time / 3600.0) * self.hardware.cost_per_hour * gpu_count

        network_util = self._network_utilization(comm_breakdown, step_time, gpu_count)

        # Build a SimulationResult with training-specific metadata
        sim_config = SimulationConfig(
            hardware=self.hardware,
            model=self.model,
            workload=WorkloadSpec(name=cfg.strategy, max_steps=1, avg_steps=1.0, step_std=0.0, context_limit=cfg.sequence_length),
            harness=AgentHarnessSpec(name="training", concurrency=gpu_count),
            target_context_tokens=cfg.sequence_length,
            precision=self.precision,
            optimization=OptimizationConfig(name="training"),
        )

        return SimulationResult(
            config=sim_config,
            latency_seconds=total_time,
            wall_time_seconds=total_time,
            tokens_total=cfg.total_tokens,
            tokens_input=cfg.total_tokens,
            tokens_output=0,
            peak_kv_gb=0.0,
            memory_required_gb=memory_per_gpu * gpu_count,
            gpu_count=gpu_count,
            feasible=memory_per_gpu <= self.hardware.memory_gb * FEASIBILITY_MAX_MEMORY_UTIL,
            bottleneck=bottleneck,
            cost_usd=cost_usd,
            utilization_gpu=mfu,
            metadata={
                "step_time_seconds": step_time,
                "compute_time_seconds": compute_time,
                "communication_time_seconds": communication_time,
                "communication_breakdown_seconds": comm_breakdown,
                "network_utilization": network_util,
                "total_flops": total_flops,
                "mfu": mfu,
                "memory_per_gpu_gb": memory_per_gpu,
                "steps": cfg.steps,
                "strategy": cfg.strategy,
                "cluster": self.cluster.name if self.cluster else None,
                "node_count": self._node_count(),
                "gpus_per_node": self._gpus_per_node(),
            },
        )


def run_training(
    model: ModelSpec,
    hardware: HardwareSpec,
    training_config: TrainingConfig,
    precision: str = "FP8",
    cluster: ClusterSpec = None,
) -> SimulationResult:
    """Convenience function to run a training simulation."""
    return TrainingEngine(model, hardware, training_config, precision, cluster).run()
