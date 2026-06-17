"""Algorithm family presets."""

from .base import AlgorithmFamily

DENSE = AlgorithmFamily(
    name="dense",
    attention_complexity="quadratic",
    has_kv_cache=True,
    kv_scaling="per_token",
    flops_per_token_forward_multiplier=1.0,
    flops_per_token_backward_multiplier=2.0,
    activation_overhead_factor=1.0,
    notes="Standard dense transformer with O(n^2) attention and GQA-style KV cache.",
)

MOE = AlgorithmFamily(
    name="moe",
    attention_complexity="quadratic",
    has_kv_cache=True,
    kv_scaling="per_token",
    flops_per_token_forward_multiplier=1.0,
    flops_per_token_backward_multiplier=2.0,
    activation_overhead_factor=1.1,
    notes="Mixture-of-Experts: active parameters used for FLOPs, total parameters for weights.",
)

MAMBA = AlgorithmFamily(
    name="mamba",
    attention_complexity="linear",
    has_kv_cache=False,
    kv_scaling="none",
    flops_per_token_forward_multiplier=1.0,
    flops_per_token_backward_multiplier=2.0,
    activation_overhead_factor=1.2,
    notes="State Space Model (Mamba): O(n) recurrent scan, no KV cache.",
)

LINEAR_ATTENTION = AlgorithmFamily(
    name="linear_attention",
    attention_complexity="linear",
    has_kv_cache=True,
    kv_scaling="compressed",
    flops_per_token_forward_multiplier=1.0,
    flops_per_token_backward_multiplier=2.0,
    activation_overhead_factor=1.1,
    default_kv_compression_ratio=0.25,
    notes="Kernelized linear attention: O(n) complexity, compressed KV state.",
)

MLA = AlgorithmFamily(
    name="mla",
    attention_complexity="quadratic",
    has_kv_cache=True,
    kv_scaling="compressed",
    flops_per_token_forward_multiplier=1.0,
    flops_per_token_backward_multiplier=2.0,
    activation_overhead_factor=1.05,
    default_kv_compression_ratio=0.025,
    notes="Multi-Head Latent Attention (DeepSeek-V3 style): ~40x KV cache reduction vs dense GQA.",
)

RING_ATTENTION = AlgorithmFamily(
    name="ring_attention",
    attention_complexity="quadratic",
    has_kv_cache=True,
    kv_scaling="chunk",
    flops_per_token_forward_multiplier=1.0,
    flops_per_token_backward_multiplier=2.0,
    activation_overhead_factor=1.3,
    notes="Ring Attention with sequence parallelism: KV distributed across devices in chunks.",
)

ALGORITHM_FAMILIES = [DENSE, MOE, MAMBA, LINEAR_ATTENTION, MLA, RING_ATTENTION]

__all__ = [
    "DENSE",
    "MOE",
    "MAMBA",
    "LINEAR_ATTENTION",
    "MLA",
    "RING_ATTENTION",
    "ALGORITHM_FAMILIES",
]
