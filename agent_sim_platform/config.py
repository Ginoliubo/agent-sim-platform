"""Default configuration, thresholds, and optimization presets."""

from .data_models import OptimizationConfig

# ---------------------------------------------------------------------------
# Bottleneck thresholds (mirrors analyze_trace.py)
# ---------------------------------------------------------------------------

BOTTLENECK_THRESHOLDS = {
    "gpu_utilization": {"critical": 0.20, "warning": 0.40},
    "gpu_memory": {"critical": 0.95, "warning": 0.85},
    "ttft": {"critical": 300.0, "warning": 180.0},  # seconds
    "context_growth": {"critical": 10000.0, "warning": 5000.0},  # tokens/step
    "io_amplification": {"critical": 10.0, "warning": 6.0},  # tool_time / llm_time
    "sandbox_cpu": {"warning": 0.05},  # control CPU below 5%
    "cost_per_task": {"critical": 500.0, "warning": 200.0},  # USD
}

# ---------------------------------------------------------------------------
# Feasibility thresholds
# ---------------------------------------------------------------------------

FEASIBILITY_MEMORY_OVERHEAD = 1.15  # weights + KV + activation + communication
FEASIBILITY_MAX_MEMORY_UTIL = 0.95

# ---------------------------------------------------------------------------
# Default utilization assumptions
# ---------------------------------------------------------------------------

DEFAULT_PREFILL_UTILIZATION = 0.95
DEFAULT_DECODE_UTILIZATION = 0.50
DEFAULT_PREFILL_ATTENTION_HBM_PASSES = 200.0  # ceiling; effective passes decay with seq_len
DEFAULT_PREFILL_SATURATION_TOKENS = 5000.0  # seq_len at which prefill util and attention reach half max
DEFAULT_PREFILL_LATENCY_FLOOR_MS = 0.0  # per-request system latency floor

# ---------------------------------------------------------------------------
# Optimization layer presets
# ---------------------------------------------------------------------------

OPT_BASELINE = OptimizationConfig(name="baseline")

OPT_LAYER1 = OptimizationConfig(
    name="layer1",
    prefix_caching=True,
    prefix_caching_hit_rate=0.80,
    flash_attention_speedup=2.0,
    spec_decode_speedup=1.5,
    continuous_batching_efficiency=1.2,
)

OPT_LAYER2 = OptimizationConfig(
    name="layer2",
    prefix_caching=True,
    prefix_caching_hit_rate=0.85,
    flash_attention_speedup=2.5,
    spec_decode_speedup=2.0,
    continuous_batching_efficiency=1.3,
    kv_compression_ratio=0.5,
    hbf_offload=True,
    hbf_offload_ratio=0.70,
    hbf_bw_gb_s=200.0,
)

OPT_LAYER3 = OptimizationConfig(
    name="layer3",
    prefix_caching=True,
    prefix_caching_hit_rate=0.90,
    flash_attention_speedup=3.0,
    spec_decode_speedup=3.0,
    continuous_batching_efficiency=1.5,
    kv_compression_ratio=0.2,
    hbf_offload=True,
    hbf_offload_ratio=0.85,
    hbf_bw_gb_s=400.0,
)

OPTIMIZATION_PRESETS = {
    "baseline": OPT_BASELINE,
    "layer1": OPT_LAYER1,
    "layer2": OPT_LAYER2,
    "layer3": OPT_LAYER3,
}

# ---------------------------------------------------------------------------
# Cost defaults
# ---------------------------------------------------------------------------

DEFAULT_DOLLAR_PER_KWH = 0.12

# ---------------------------------------------------------------------------
# Workload defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS = 100
DEFAULT_CONTEXT_LIMIT = 128000
