"""Core data models for the agent simulation platform.

These dataclasses represent the unified abstraction across hardware, models,
workloads, harness configurations, optimizations, and simulation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .algorithms.base import AlgorithmFamily


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareSpec:
    """Immutable specification of an AI accelerator (GPU, NPU, TPU).

    All bandwidth/throughput numbers use SI-style decimal prefixes unless noted.
    - Memory: GB, TB/s
    - Compute: TFLOPS
    - Interconnect: GB/s
    - Power: W
    - Cost: USD/hour
    """

    name: str
    vendor: str  # e.g. "nvidia", "huawei", "google"
    kind: str  # "gpu" | "npu" | "tpu"
    memory_gb: float
    memory_bw_tb_s: float
    fp16_tflops: float
    fp8_tflops: float
    fp4_tflops: float = 0.0
    interconnect_bw_gb_s: float = 0.0  # NVLink / HCCS / ICI per link or aggregate
    pcie_bw_gb_s: float = 0.0
    power_w: float = 0.0
    cost_per_hour: float = 0.0
    release_year: int = 0
    is_future: bool = False
    uncertainty_range: float = 0.0  # ±% for speculative future specs
    notes: str = ""

    def memory_bw_bytes_s(self) -> float:
        """HBM memory bandwidth in bytes per second."""
        return self.memory_bw_tb_s * 1e12

    def effective_flops(self, precision: str, utilization: float = 1.0) -> float:
        """Effective compute in FLOPS at a given precision and utilization."""
        precision = precision.upper()
        base = {
            "FP32": self.fp16_tflops / 2.0,
            "FP16": self.fp16_tflops,
            "BF16": self.fp16_tflops,
            "FP8": self.fp8_tflops,
            "INT8": self.fp8_tflops,
            "FP4": self.fp4_tflops,
            "INT4": self.fp4_tflops,
        }.get(precision, self.fp16_tflops)
        return base * 1e12 * utilization

    def bytes_per_param(self, precision: str) -> float:
        """Bytes per parameter for a given precision string."""
        return BYTES_PER_PARAM[precision.upper()]

    def __str__(self) -> str:
        flag = " [future]" if self.is_future else ""
        return f"{self.name} ({self.vendor} {self.kind}){flag}"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification of a language model."""

    name: str
    total_params_b: float
    active_params_b: float
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    num_experts: int = 1
    top_k: int = 1
    vocab_size: int = 150000
    architecture: str = "dense"  # "dense" | "moe"
    context_len_default: int = 32768
    algorithm_family: Optional[AlgorithmFamily] = None

    def __post_init__(self):
        # Backward compatibility: if algorithm_family not set, infer from architecture
        if self.algorithm_family is None:
            from .algorithms.families import DENSE, MOE

            object.__setattr__(
                self, "algorithm_family", MOE if self.is_moe else DENSE
            )

    @property
    def is_moe(self) -> bool:
        return self.architecture == "moe" or self.num_experts > 1

    def bytes_per_param(self, precision: str) -> float:
        """Bytes per parameter at the given precision."""
        return BYTES_PER_PARAM[precision.upper()]

    def weight_memory_gb(self, precision: str = "FP8") -> float:
        """Total weight memory in GB."""
        bytes_per_param = BYTES_PER_PARAM[precision.upper()]
        return self.total_params_b * 1e9 * bytes_per_param / (1024**3)

    def active_weight_memory_gb(self, precision: str = "FP8") -> float:
        """Active weight memory in GB (MoE routing path)."""
        bytes_per_param = BYTES_PER_PARAM[precision.upper()]
        return self.active_params_b * 1e9 * bytes_per_param / (1024**3)

    def kv_bytes_per_token(self, kv_precision: str = "FP8") -> float:
        """KV cache bytes per token, delegated to algorithm family."""
        if self.algorithm_family is None:
            # Fallback to dense transformer heuristic
            n_kv_heads = max(1, self.n_heads // 4)
            kv_per_token = 2 * self.n_layers * n_kv_heads * self.d_head
            return kv_per_token * BYTES_PER_PARAM[kv_precision.upper()]
        return self.algorithm_family.kv_bytes_per_token(self, kv_precision)

    def flops_per_token_forward(self) -> float:
        """FLOPs for one forward pass token."""
        if self.algorithm_family is None:
            return 2 * self.active_params_b * 1e9
        return self.algorithm_family.flops_per_token_forward(self)

    def flops_per_token_backward(self) -> float:
        """FLOPs for one backward pass token."""
        if self.algorithm_family is None:
            return 4 * self.active_params_b * 1e9
        return self.algorithm_family.flops_per_token_backward(self)

    def flops_per_token_training(self) -> float:
        """FLOPs for one training token (forward + backward)."""
        return self.flops_per_token_forward() + self.flops_per_token_backward()

    def gpu_needed(self, memory_gb: float, precision: str = "FP8", overhead: float = 1.15) -> int:
        """Minimum accelerator count to fit weights given per-device memory."""
        total_bytes = self.weight_memory_gb(precision) * (1024**3) * overhead
        per_device = memory_gb * (1024**3)
        return max(1, int(total_bytes // per_device) + (1 if total_bytes % per_device else 0))


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

@dataclass
class WorkloadSpec:
    """Stochastic specification of an agent workload."""

    name: str
    max_steps: int
    avg_steps: float
    step_std: float
    context_limit: int
    token_distributions: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "thought": (350.0, 100.0),
            "action": (120.0, 50.0),
            "observation": (800.0, 600.0),
        }
    )
    tool_delays: Dict[str, float] = field(
        default_factory=lambda: {
            "open": 0.05,
            "view": 0.03,
            "edit": 0.10,
            "bash_test": 8.0,
            "bash_install": 25.0,
            "search": 0.5,
            "submit": 0.01,
        }
    )
    tool_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "open": 0.30,
            "view": 0.22,
            "bash_test": 0.15,
            "bash_install": 0.05,
            "edit": 0.16,
            "search": 0.08,
            "submit": 0.04,
        }
    )
    env_init_delay: float = 60.0
    description: str = ""


# ---------------------------------------------------------------------------
# Agent Harness
# ---------------------------------------------------------------------------

@dataclass
class AgentHarnessSpec:
    """Configuration of the agent software harness and runtime topology."""

    name: str
    concurrency: int = 1
    control_cpu_percent: float = 0.05  # 5% of a control CPU core per GPU node
    agent_cpu_peak_cores: int = 8
    compile_test_cores: int = 32
    sandbox_type: str = "docker"
    compaction_strategy: Optional[str] = None  # e.g. "sliding-window", "hierarchy", "summarize"
    sub_agent_enabled: bool = False
    sub_agent_count: int = 0
    description: str = ""


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """Stack of inference optimization techniques."""

    name: str = "baseline"
    prefix_caching: bool = False
    prefix_caching_hit_rate: float = 0.0
    flash_attention_speedup: float = 1.0
    spec_decode_speedup: float = 1.0
    continuous_batching_efficiency: float = 1.0
    kv_compression_ratio: float = 1.0  # 1.0 = no compression; 0.1 = 10x compression
    hbf_offload: bool = False
    hbf_offload_ratio: float = 0.0
    hbf_bw_gb_s: float = 200.0
    quantization_bits: int = 8  # model weight quantization bits

    def effective_prefill_speedup(self) -> float:
        """Combined prefill speedup factor."""
        pc = 1.0
        if self.prefix_caching and self.prefix_caching_hit_rate > 0:
            pc = 1.0 / (1.0 - self.prefix_caching_hit_rate)
        return pc * self.flash_attention_speedup * self.continuous_batching_efficiency

    def effective_decode_speedup(self) -> float:
        """Combined decode speedup factor."""
        return self.spec_decode_speedup * self.continuous_batching_efficiency

    def kv_bytes_per_token(self, base_kv_bytes: float) -> float:
        """KV bytes after compression."""
        return base_kv_bytes * self.kv_compression_ratio


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------

@dataclass
class ParallelismConfig:
    """Distributed parallelism configuration for training."""

    dp: int = 1   # data parallel
    tp: int = 1   # tensor parallel
    pp: int = 1   # pipeline parallel
    ep: int = 1   # expert parallel (MoE)
    sp: int = 1   # sequence parallel

    @property
    def total_gpus(self) -> int:
        return self.dp * self.tp * self.pp * self.ep


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for a training job."""

    strategy: str = "pretrain"  # pretrain | sft | rlhf | dpo | grpo
    dataset_tokens: int = 0
    epochs: int = 1
    global_batch_size: int = 1024
    micro_batch_size: int = 1
    sequence_length: int = 4096
    optimizer: str = "adamw"    # adamw | sgd
    gradient_checkpointing: bool = True
    zero_stage: int = 1         # 0/1/2/3
    mfu_target: float = 0.35    # model FLOPs utilization target
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    data_loading_overhead_fraction: float = 0.05
    notes: str = ""

    @property
    def total_tokens(self) -> int:
        return int(self.dataset_tokens * self.epochs)

    @property
    def steps(self) -> int:
        tokens_per_step = self.global_batch_size * self.sequence_length
        return max(1, self.total_tokens // tokens_per_step)


# ---------------------------------------------------------------------------
# Inference Service
# ---------------------------------------------------------------------------

@dataclass
class InferenceServiceConfig:
    """Configuration for an inference serving workload."""

    arrival_rate_per_sec: float = 1.0
    arrival_distribution: str = "poisson"  # poisson | bursty | fixed
    target_ttft_ms: float = 2000.0
    target_tpot_ms: float = 50.0
    max_batch_size: int = 128
    max_queue_len: int = 64
    prefill_decode_disaggregation: bool = False
    request_length_mean: int = 4096
    request_length_std: int = 2048
    output_length_mean: int = 512
    output_length_std: int = 256
    simulation_duration_seconds: float = 60.0


# ---------------------------------------------------------------------------
# Simulation Config / Result
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Complete input configuration for a simulation run."""

    hardware: HardwareSpec
    model: ModelSpec
    workload: WorkloadSpec
    harness: AgentHarnessSpec
    target_context_tokens: int = 32768
    precision: str = "FP8"
    kv_precision: str = "FP8"
    tp: int = 8
    pp: int = 1
    batch_size: int = 1
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    random_seed: int = 42


@dataclass
class SimulationResult:
    """Unified output from a simulation run."""

    config: SimulationConfig
    latency_seconds: float = 0.0
    wall_time_seconds: float = 0.0
    tokens_total: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    peak_kv_gb: float = 0.0
    memory_required_gb: float = 0.0
    gpu_count: int = 1
    feasible: bool = False
    bottleneck: str = ""
    cost_usd: float = 0.0
    utilization_gpu: float = 0.0
    per_step: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize result to a plain dictionary."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationResult":
        """Deserialize a SimulationResult from a plain dictionary."""
        config_data = data.get("config", {})

        def _build_hardware(hw: Dict) -> HardwareSpec:
            return HardwareSpec(**hw)

        def _build_model(m: Dict) -> ModelSpec:
            return ModelSpec(**m)

        def _build_workload(w: Dict) -> WorkloadSpec:
            return WorkloadSpec(**w)

        def _build_harness(h: Dict) -> AgentHarnessSpec:
            return AgentHarnessSpec(**h)

        def _build_optimization(o: Dict) -> OptimizationConfig:
            return OptimizationConfig(**o)

        def _build_config(c: Dict) -> SimulationConfig:
            return SimulationConfig(
                hardware=_build_hardware(c.get("hardware", {})),
                model=_build_model(c.get("model", {})),
                workload=_build_workload(c.get("workload", {})),
                harness=_build_harness(c.get("harness", {})),
                target_context_tokens=c.get("target_context_tokens", 32768),
                precision=c.get("precision", "FP8"),
                kv_precision=c.get("kv_precision", "FP8"),
                tp=c.get("tp", 8),
                pp=c.get("pp", 1),
                batch_size=c.get("batch_size", 1),
                optimization=_build_optimization(c.get("optimization", {})),
                random_seed=c.get("random_seed", 42),
            )

        result_data = dict(data)
        result_data["config"] = _build_config(config_data)
        return cls(**result_data)


# ---------------------------------------------------------------------------
# Benchmark & Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCase:
    """An authoritative benchmark fixture for calibration.

    A fixture pairs a known input configuration with observed metrics from an
    external source (paper, tech report, or reproducible benchmark).
    """

    name: str
    domain: str  # "training" | "serving" | "capacity"
    source: str
    source_url: str = ""
    hardware_names: List[str] = field(default_factory=list)
    model_name: str = ""
    algorithm_name: str = "dense"
    config: Dict = field(default_factory=dict)
    observed_metrics: Dict = field(default_factory=dict)
    tolerance: Dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class CalibrationConfig:
    """Configuration for the calibration / auto-fitting engine."""

    domain: str = "all"  # "training" | "serving" | "capacity" | "all"
    metric: str = "mape"  # "mape" | "rmse" | "combined"
    fit_params: List[str] = field(
        default_factory=lambda: [
            "mfu_target",
            "default_prefill_utilization",
            "default_decode_utilization",
            "activation_overhead_factor",
            "continuous_batching_efficiency",
        ]
    )
    max_iterations: int = 100
    tolerance: float = 0.05


@dataclass
class ProfilingReport:
    """Multi-layer profiling report for a simulation or trace."""

    layers: Dict[str, Dict] = field(default_factory=dict)
    correlations: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    raw_samples: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BYTES_PER_PARAM: Dict[str, float] = {
    "FP32": 4.0,
    "FP16": 2.0,
    "BF16": 2.0,
    "FP8": 1.0,
    "INT8": 1.0,
    "INT4": 0.5,
    "FP4": 0.5,
}

PRECISION_NORMALIZE: Dict[str, str] = {
    "fp32": "FP32",
    "fp16": "FP16",
    "bf16": "BF16",
    "fp8": "FP8",
    "int8": "INT8",
    "int4": "INT4",
    "fp4": "FP4",
}


def normalize_precision(precision: str) -> str:
    """Normalize precision string to canonical form."""
    return PRECISION_NORMALIZE.get(precision.lower(), precision.upper())
