"""Algorithm family base types and behavior definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_models import ModelSpec


@dataclass(frozen=True)
class AlgorithmFamily:
    """Immutable specification of a model algorithm family.

    Defines how FLOPs, KV cache, and memory scale for a given architecture.
    """

    name: str
    attention_complexity: str  # "quadratic" | "linear" | "subquadratic"
    has_kv_cache: bool
    kv_scaling: str  # "per_token" | "chunk" | "none" | "compressed"
    flops_per_token_forward_multiplier: float = 1.0  # relative to 2 * P_active
    flops_per_token_backward_multiplier: float = 2.0  # relative to 2 * P_active
    activation_overhead_factor: float = 1.0
    default_kv_compression_ratio: float = 1.0
    notes: str = ""

    def kv_bytes_per_token(self, model: "ModelSpec", kv_precision: str = "FP8") -> float:
        """KV cache bytes per token for this algorithm family."""
        from ..data_models import BYTES_PER_PARAM

        if not self.has_kv_cache:
            return 0.0

        bytes_per_param = BYTES_PER_PARAM[kv_precision.upper()]

        if self.kv_scaling == "none":
            return 0.0

        if self.kv_scaling == "compressed":
            # MLA-style aggressive compression; use default compression ratio
            n_kv_heads = max(1, model.n_heads // 8)
            kv_per_token = 2 * model.n_layers * n_kv_heads * model.d_head
            return kv_per_token * bytes_per_param * self.default_kv_compression_ratio

        if self.kv_scaling == "chunk":
            # Ring Attention / sequence-parallel chunking: same per-token KV, distributed
            n_kv_heads = max(1, model.n_heads // 4)
            kv_per_token = 2 * model.n_layers * n_kv_heads * model.d_head
            return kv_per_token * bytes_per_param

        # Default "per_token": standard transformer GQA heuristic
        n_kv_heads = max(1, model.n_heads // 4)
        kv_per_token = 2 * model.n_layers * n_kv_heads * model.d_head
        return kv_per_token * bytes_per_param

    def flops_per_token_forward(self, model: "ModelSpec") -> float:
        """FLOPs for one forward pass token."""
        active_params = model.active_params_b * 1e9
        return 2 * active_params * self.flops_per_token_forward_multiplier

    def flops_per_token_backward(self, model: "ModelSpec") -> float:
        """FLOPs for one backward pass token (typically 2x forward)."""
        forward_flops = self.flops_per_token_forward(model)
        return forward_flops * self.flops_per_token_backward_multiplier

    def flops_per_token_training(self, model: "ModelSpec") -> float:
        """FLOPs for one training token (forward + backward)."""
        return self.flops_per_token_forward(model) + self.flops_per_token_backward(model)

    def __str__(self) -> str:
        return self.name


__all__ = ["AlgorithmFamily"]
