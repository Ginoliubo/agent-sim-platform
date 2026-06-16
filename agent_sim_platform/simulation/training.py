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
    """

    def __init__(
        self,
        model: ModelSpec,
        hardware: HardwareSpec,
        training_config: TrainingConfig,
        precision: str = "FP8",
    ):
        self.model = model
        self.hardware = hardware
        self.training_config = training_config
        self.precision = precision.upper()
        self.bytes_per_param = model.bytes_per_param(self.precision)

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

    def _communication_time_per_step(
        self, gpu_count: int, compute_time: float
    ) -> float:
        """Estimate communication time per step."""
        cfg = self.training_config
        parallelism = cfg.parallelism
        model_size_bytes = self.model.total_params_b * 1e9 * self.bytes_per_param

        comm_time = 0.0

        # Data parallel gradient All-Reduce
        if parallelism.dp > 1 and cfg.zero_stage < 3:
            # Ring all-reduce: 2*(N-1)/N * model_size
            bytes_moved = 2 * (parallelism.dp - 1) / parallelism.dp * model_size_bytes
            # Use interconnect bandwidth if cross-node, otherwise NVLink
            bw = self.hardware.interconnect_bw_gb_s * 1e9
            if bw <= 0:
                bw = self.hardware.pcie_bw_gb_s * 1e9
            if bw > 0:
                comm_time += bytes_moved / bw

        # Tensor parallel all-reduce per layer
        if parallelism.tp > 1:
            layer_size_bytes = model_size_bytes / self.model.n_layers
            bytes_per_layer = 2 * layer_size_bytes  # two all-reduces per layer
            bw = self.hardware.interconnect_bw_gb_s * 1e9
            if bw <= 0:
                bw = self.hardware.memory_bw_bytes_s()  # fallback
            if bw > 0:
                comm_time += self.model.n_layers * bytes_per_layer / bw

        # Pipeline parallel bubble
        if parallelism.pp > 1:
            # Bubble ratio: (PP-1)/(num_micro_batches + PP - 1)
            num_micro_batches = max(
                parallelism.pp * 4,
                cfg.global_batch_size // (cfg.micro_batch_size * parallelism.dp),
            )
            bubble_ratio = (parallelism.pp - 1) / (num_micro_batches + parallelism.pp - 1)
            comm_time += compute_time * bubble_ratio

        # MoE expert parallel all-to-all
        if self.model.is_moe and parallelism.ep > 1:
            hidden_bytes = (
                cfg.global_batch_size
                * cfg.sequence_length
                * self.model.d_model
                * self.bytes_per_param
            )
            bytes_moved = 2 * (parallelism.ep - 1) / parallelism.ep * hidden_bytes
            bw = self.hardware.interconnect_bw_gb_s * 1e9
            if bw > 0:
                comm_time += bytes_moved / bw

        return comm_time

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
        communication_time = self._communication_time_per_step(gpu_count, compute_time)
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
            bottleneck = "communication: interconnect bottleneck"
        elif data_loading_time > compute_time * 0.2:
            bottleneck = "data-loading: IO bottleneck"
        else:
            bottleneck = "compute: GPU-bound"

        cost_usd = (total_time / 3600.0) * self.hardware.cost_per_hour * gpu_count

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
                "total_flops": total_flops,
                "mfu": mfu,
                "memory_per_gpu_gb": memory_per_gpu,
                "steps": cfg.steps,
                "strategy": cfg.strategy,
            },
        )


def run_training(
    model: ModelSpec,
    hardware: HardwareSpec,
    training_config: TrainingConfig,
    precision: str = "FP8",
) -> SimulationResult:
    """Convenience function to run a training simulation."""
    return TrainingEngine(model, hardware, training_config, precision).run()
