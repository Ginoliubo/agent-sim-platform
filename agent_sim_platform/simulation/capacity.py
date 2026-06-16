"""Capacity estimator: memory, latency, and throughput for a model+context."""

from ..config import DEFAULT_DECODE_UTILIZATION, DEFAULT_PREFILL_UTILIZATION, FEASIBILITY_MAX_MEMORY_UTIL
from ..data_models import HardwareSpec, ModelSpec, SimulationResult
from ..utils.units import gb_to_bytes


class CapacityEstimator:
    """Estimate whether a model+context fits on a hardware configuration."""

    def __init__(
        self,
        model: ModelSpec,
        hardware: HardwareSpec,
        precision: str = "FP8",
        kv_precision: str = "FP8",
        tp: int = 8,
        pp: int = 1,
        batch_size: int = 1,
    ):
        self.model = model
        self.hardware = hardware
        self.precision = precision.upper()
        self.kv_precision = kv_precision.upper()
        self.tp = tp
        self.pp = pp
        self.batch_size = batch_size
        self.total_gpus = tp * pp

    def weight_memory_gb(self) -> float:
        """Model weight memory in GB."""
        return self.model.weight_memory_gb(self.precision)

    def kv_memory_gb(self, context_tokens: int) -> float:
        """KV cache memory for a given context length in GB."""
        kv_bytes_per_token = self.model.kv_bytes_per_token(self.kv_precision)
        return context_tokens * self.batch_size * kv_bytes_per_token / gb_to_bytes(1.0)

    def activation_memory_gb(self) -> float:
        """Activation memory (simplified, bs=1 FlashAttention)."""
        # Conservative 5 GB per GPU for bs=1 large model
        return 5.0 * self.total_gpus

    def total_memory_gb(self, context_tokens: int) -> float:
        """Total memory required in GB."""
        return (
            self.weight_memory_gb()
            + self.kv_memory_gb(context_tokens)
            + self.activation_memory_gb()
        )

    def fits(self, context_tokens: int) -> bool:
        """Check if configuration fits in available HBM."""
        available = self.hardware.memory_gb * self.total_gpus * FEASIBILITY_MAX_MEMORY_UTIL
        return self.total_memory_gb(context_tokens) <= available

    def prefill_latency_seconds(self, context_tokens: int) -> float:
        """Time to prefill a prompt of given length."""
        flops = 2 * self.model.active_params_b * 1e9 * context_tokens
        effective_flops = (
            self.hardware.effective_flops(self.precision, DEFAULT_PREFILL_UTILIZATION)
            * self.total_gpus
        )
        return flops / effective_flops

    def decode_latency_per_token_seconds(self) -> float:
        """Decode latency for one token."""
        flops = 2 * self.model.active_params_b * 1e9
        effective_compute = (
            self.hardware.effective_flops(self.precision, DEFAULT_DECODE_UTILIZATION)
            * self.total_gpus
        )
        compute_time = flops / effective_compute

        bytes_per_token = (
            self.model.active_params_b * 1e9 * self.model.bytes_per_param(self.precision)
            + self.model.kv_bytes_per_token(self.kv_precision)
        )
        effective_bw = self.hardware.memory_bw_bytes_s() * DEFAULT_DECODE_UTILIZATION * self.total_gpus
        memory_time = bytes_per_token / effective_bw

        return max(compute_time, memory_time)

    def throughput_tokens_per_second(self) -> float:
        """Sustained decode throughput in tokens/s."""
        return 1.0 / self.decode_latency_per_token_seconds()

    def estimate(self, context_tokens: int) -> SimulationResult:
        """Return a capacity-focused SimulationResult."""
        from ..data_models import (
            AgentHarnessSpec,
            OptimizationConfig,
            SimulationConfig,
            WorkloadSpec,
        )

        total_memory = self.total_memory_gb(context_tokens)
        fits = self.fits(context_tokens)
        bottleneck = "none" if fits else "memory: HBM overflow"

        return SimulationResult(
            config=SimulationConfig(
                hardware=self.hardware,
                model=self.model,
                workload=WorkloadSpec(
                    name="capacity",
                    max_steps=1,
                    avg_steps=1.0,
                    step_std=0.0,
                    context_limit=self.model.context_len_default,
                ),
                harness=AgentHarnessSpec(
                    name="capacity",
                    concurrency=1,
                    control_cpu_percent=0.0,
                    agent_cpu_peak_cores=0,
                    compile_test_cores=0,
                ),
                target_context_tokens=context_tokens,
                precision=self.precision,
                kv_precision=self.kv_precision,
                tp=self.tp,
                pp=self.pp,
                batch_size=self.batch_size,
                optimization=OptimizationConfig(name="baseline"),
            ),
            latency_seconds=self.prefill_latency_seconds(context_tokens),
            tokens_total=context_tokens,
            tokens_input=context_tokens,
            tokens_output=0,
            peak_kv_gb=self.kv_memory_gb(context_tokens),
            memory_required_gb=total_memory,
            gpu_count=self.total_gpus,
            feasible=fits,
            bottleneck=bottleneck,
            cost_usd=0.0,
            utilization_gpu=0.0,
            metadata={
                "prefill_latency_s": self.prefill_latency_seconds(context_tokens),
                "decode_latency_per_token_s": self.decode_latency_per_token_seconds(),
                "decode_tps": self.throughput_tokens_per_second(),
                "weight_memory_gb": self.weight_memory_gb(),
                "activation_memory_gb": self.activation_memory_gb(),
            },
        )
