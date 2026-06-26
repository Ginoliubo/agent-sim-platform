"""Inference serving simulation engine with continuous batching.

Uses a discrete-event simulation for performance:
- request arrivals
- prefill batch completions
- decode token completions

Supports:
- Distributed deployment via TP/PP/CP
- Prefill-Decode (PD) disaggregation
- Attention-FFN-Decode (AFD) disaggregation (v1)
- Tiered KV-cache offload (HBM/DRAM/SSD/ICMS/CXL)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config as sim_config
from ..calibration.residual import ResidualModel
from ..config import FEASIBILITY_MAX_MEMORY_UTIL
from ..data_models import (
    AFDConfig,
    AgentHarnessSpec,
    HardwareSpec,
    InferenceServiceConfig,
    ModelSpec,
    OptimizationConfig,
    PDConfig,
    SimulationConfig,
    SimulationResult,
    WorkloadSpec,
)
from ..hardware import ClusterSpec
from ..utils.units import gb_to_bytes


@dataclass
class CommunicationBreakdown:
    """Network communication cost breakdown for one distributed inference step."""

    tp_bytes_per_token: float = 0.0
    pp_bytes_per_token: float = 0.0
    cp_bytes_per_token: float = 0.0
    tp_time_per_token_ms: float = 0.0
    pp_time_per_token_ms: float = 0.0
    cp_time_per_token_ms: float = 0.0
    kv_transfer_time_ms: float = 0.0
    kv_offload_time_ms: float = 0.0
    afd_transfer_time_ms: float = 0.0


@dataclass
class Request:
    """A single inference request."""

    id: int
    arrival_time: float
    input_len: int
    output_len: int
    start_prefill_time: float = -1.0
    first_token_time: float = -1.0
    completion_time: float = -1.0
    generated_tokens: int = 0
    dropped: bool = False
    kv_transferred: bool = False

    @property
    def ttft(self) -> float:
        if self.first_token_time < 0:
            return -1.0
        return self.first_token_time - self.arrival_time

    @property
    def total_latency(self) -> float:
        if self.completion_time < 0:
            return -1.0
        return self.completion_time - self.arrival_time


class InferenceServingEngine:
    """Simulate an inference serving system with continuous batching.

    Supports distributed deployment via tensor parallelism (TP), pipeline
    parallelism (PP), and context/sequence parallelism (CP). Models intra- and
    inter-node communication overheads for prefill and decode.

    Additionally supports PD disaggregation, AFD disaggregation, and tiered
    KV-cache offload.
    """

    def __init__(
        self,
        model: ModelSpec,
        hardware: HardwareSpec,
        service_config: InferenceServiceConfig,
        precision: str = "FP8",
        kv_precision: str = "FP8",
        optimization: OptimizationConfig = None,
        gpu_count: int = 1,
        cluster: ClusterSpec = None,
        tp: int = 1,
        pp: int = 1,
        cp: int = 1,
        seed: int = 42,
    ):
        self.model = model
        self.hardware = hardware
        self.service_config = service_config
        self.precision = precision.upper()
        self.kv_precision = kv_precision.upper()
        self.optimization = optimization or OptimizationConfig()
        self.cluster = cluster
        self.tp = tp
        self.pp = pp
        self.cp = cp
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.bytes_per_param = model.bytes_per_param(self.precision)
        self.kv_bytes_per_token = model.kv_bytes_per_token(self.kv_precision)
        # MoE inference only loads active experts; use active weight footprint.
        self.weight_memory_gb = (
            model.active_weight_memory_gb(self.precision)
            if model.is_moe
            else model.weight_memory_gb(self.precision)
        )

        # PD / AFD configs
        self.pd_config = service_config.pd_config or PDConfig()
        self.afd_config = service_config.afd_config or AFDConfig()
        # Backward compat: simple boolean enables PD with equal split
        if service_config.prefill_decode_disaggregation and not self.pd_config.enabled:
            self.pd_config = PDConfig(
                enabled=True,
                prefill_gpu_count=max(1, gpu_count // 2),
                decode_gpu_count=max(1, gpu_count // 2),
            )

        # Resolve effective GPU count and parallelism
        self._resolve_parallelism(gpu_count)

        # Pool-level concurrency for PD/AFD disaggregation.
        # Each (tp*pp*cp) group can run one request at a time; data-parallel
        # instances are derived from the pool size.
        group_size = max(1, self.tp * self.pp * self.cp)
        self.prefill_instances = max(1, self._prefill_gpu_count() // group_size)
        self.decode_instances = max(1, self._decode_gpu_count() // group_size)

        # Dynamic KV offload hit rates based on expected working set
        self.optimization.configure_kv_hit_rates(
            context_tokens=self.service_config.request_length_mean,
            batch_size=max(1, self.service_config.max_batch_size),
            kv_bytes_per_token=self.kv_bytes_per_token,
        )

        # Derived timing constants
        self.prefill_time_per_token = self._compute_prefill_time_per_token()
        self.kv_offload_time_per_token = self._compute_kv_offload_time_per_token()
        self.afd_transfer_time_per_token = self._compute_afd_transfer_time_per_token()
        self.decode_time_per_token = self._compute_decode_time_per_token()
        self.kv_transfer_time_per_request = self._compute_kv_transfer_time_per_request()

    def _resolve_parallelism(self, requested_gpu_count: int) -> None:
        """Determine GPU count and TP/PP/CP based on inputs and memory."""
        explicit_parallelism = self.tp * self.pp * self.cp

        # For disaggregated serving, total GPUs must cover the pools, and
        # explicit parallelism describes the per-request footprint, not the total.
        if self.pd_config.enabled:
            pool_total = self.pd_config.prefill_gpu_count + self.pd_config.decode_gpu_count
        elif self.afd_config.enabled:
            pool_total = (
                self.afd_config.attention_gpu_count
                + self.afd_config.ffn_gpu_count
                + self.afd_config.decode_gpu_count
            )
        else:
            pool_total = 0

        if pool_total > 0:
            self.gpu_count = max(pool_total, explicit_parallelism)
        elif explicit_parallelism > 1:
            self.gpu_count = explicit_parallelism
        elif self.cluster is not None:
            self.gpu_count = self.cluster.total_gpus
        else:
            self.gpu_count = max(1, requested_gpu_count)

        # PD separation: derive P/D GPU counts if not explicitly set
        if self.pd_config.enabled:
            if self.pd_config.prefill_gpu_count <= 0:
                self.pd_config = self._replace_pd(
                    prefill_gpu_count=max(1, self.gpu_count // 2)
                )
            if self.pd_config.decode_gpu_count <= 0:
                self.pd_config = self._replace_pd(
                    decode_gpu_count=max(1, self.gpu_count // 2)
                )
        else:
            self.pd_config = self._replace_pd(prefill_gpu_count=self.gpu_count, decode_gpu_count=self.gpu_count)

        # AFD separation: derive A/F/D GPU counts if not explicitly set
        if self.afd_config.enabled:
            total = self.afd_config.attention_gpu_count + self.afd_config.ffn_gpu_count + self.afd_config.decode_gpu_count
            if total <= 0:
                self.afd_config = self._replace_afd(
                    attention_gpu_count=max(1, self.gpu_count // 3),
                    ffn_gpu_count=max(1, self.gpu_count // 3),
                    decode_gpu_count=max(1, self.gpu_count // 3),
                )

        # Validate cluster capacity
        if self.cluster is not None and self.gpu_count > self.cluster.total_gpus:
            raise ValueError(
                f"Requested parallelism needs {self.gpu_count} GPUs, exceeding cluster "
                f"{self.cluster.name} capacity ({self.cluster.total_gpus})"
            )

        # Auto-scale context parallelism for AFD if the per-request footprint
        # overflows HBM and the caller did not explicitly request a CP size.
        # AFD fixtures often leave CP at default and need CP to fit long-context
        # MoE weights; PD fixtures typically specify enough parallelism already.
        if self.cp == 1 and self.afd_config.enabled:
            max_context = (
                self.service_config.request_length_mean + self.service_config.output_length_mean
            )
            min_instances = 2
            base_group = max(1, self.tp * self.pp)
            pool_size = min(self._prefill_gpu_count(), self._decode_gpu_count())
            for _ in range(6):
                if self._fits_in_cluster(
                    max_context, batch_size=self.service_config.max_batch_size
                ):
                    break
                next_cp = self.cp * 2
                if pool_size > 0 and next_cp > pool_size // (base_group * min_instances):
                    break
                self.cp = next_cp

    def _replace_pd(self, **kwargs) -> PDConfig:
        """Return a new PDConfig with updated fields."""
        data = self.pd_config.__dict__.copy()
        data.update(kwargs)
        return PDConfig(**data)

    def _replace_afd(self, **kwargs) -> AFDConfig:
        """Return a new AFDConfig with updated fields."""
        data = self.afd_config.__dict__.copy()
        data.update(kwargs)
        return AFDConfig(**data)

    def _prefill_gpu_count(self) -> int:
        if self.pd_config.enabled:
            return self.pd_config.prefill_gpu_count
        if self.afd_config.enabled:
            return self.afd_config.attention_gpu_count + self.afd_config.ffn_gpu_count
        return self.gpu_count

    def _decode_gpu_count(self) -> int:
        if self.pd_config.enabled:
            return self.pd_config.decode_gpu_count
        if self.afd_config.enabled:
            # AFD decode uses attention + FFN + decode pools in a pipelined
            # fashion; the effective decode capacity is the total pool size.
            return (
                self.afd_config.attention_gpu_count
                + self.afd_config.ffn_gpu_count
                + self.afd_config.decode_gpu_count
            )
        return self.gpu_count

    def _memory_per_gpu_gb(self, context_tokens: int, batch_size: int = 1) -> float:
        """Estimate memory required per GPU for the distributed config."""
        opt = self.optimization
        kv_bytes_per_token = self.kv_bytes_per_token * opt.kv_compression_ratio

        weight_per_gpu = self.weight_memory_gb / max(1, self.tp * self.pp)
        kv_per_gpu = (
            context_tokens * batch_size * kv_bytes_per_token / gb_to_bytes(1.0)
        ) / max(1, self.cp)
        # Activations scale with sequence and batch; sharded by TP and CP.
        activation_bytes = (
            2 * context_tokens * batch_size * self.model.d_model * self.bytes_per_param
        )
        activation_per_gpu = activation_bytes / max(1, self.tp * self.cp) / gb_to_bytes(1.0)
        # Add small per-GPU overhead for working buffers
        activation_per_gpu += 2.0 * batch_size
        return weight_per_gpu + kv_per_gpu + activation_per_gpu

    def _fits_in_cluster(self, context_tokens: int, batch_size: int = 1) -> bool:
        """Check whether the distributed config fits in cluster HBM."""
        return (
            self._memory_per_gpu_gb(context_tokens, batch_size)
            <= self.hardware.memory_gb * sim_config.FEASIBILITY_MAX_MEMORY_UTIL
        )

    def _intra_node_bw_bytes_s(self) -> float:
        """Aggregate intra-node bandwidth in bytes/s."""
        if self.cluster is None:
            return self.hardware.memory_bw_bytes_s()
        return self.cluster.topology.intra_node_bw_gb_s * 1e9

    def _inter_node_bw_bytes_s(self) -> float:
        """Effective per-node inter-node bandwidth in bytes/s."""
        if self.cluster is None:
            return self.hardware.memory_bw_bytes_s()
        return self.cluster.topology.effective_inter_node_bw_gb_s * 1e9

    def _tp_bandwidth_bytes_s(self) -> float:
        """Bandwidth available to a TP group."""
        if self.tp <= self._gpus_per_node():
            return self._intra_node_bw_bytes_s()
        return self._inter_node_bw_bytes_s()

    def _pp_bandwidth_bytes_s(self) -> float:
        """Bandwidth available between PP stages."""
        return self._inter_node_bw_bytes_s()

    def _cp_bandwidth_bytes_s(self) -> float:
        """Bandwidth available to a CP group (ring)."""
        return self._inter_node_bw_bytes_s()

    def _gpus_per_node(self) -> int:
        if self.cluster is None:
            return self.gpu_count
        return self.cluster.gpus_per_node

    def _node_count(self) -> int:
        if self.cluster is None:
            return 1
        return self.cluster.node_count

    def _communication_breakdown(self, seq_len: int, batch_size: int, phase: str = "prefill") -> CommunicationBreakdown:
        """Estimate communication bytes and time per token for prefill/decode."""
        model = self.model
        opt = self.optimization
        hidden_bytes = model.d_model * self.bytes_per_param
        n_layers = model.n_layers

        comm = CommunicationBreakdown()

        if self.tp > 1:
            activation_size = batch_size * seq_len * hidden_bytes
            bytes_per_layer = 2 * activation_size * 2
            comm.tp_bytes_per_token = bytes_per_layer * n_layers / seq_len
            bw = self._tp_bandwidth_bytes_s() * opt.continuous_batching_efficiency
            if bw > 0:
                comm.tp_time_per_token_ms = (comm.tp_bytes_per_token / bw) * 1000.0

        if self.pp > 1:
            activation_size = batch_size * seq_len * hidden_bytes
            comm.pp_bytes_per_token = activation_size * (self.pp - 1) / seq_len
            bw = self._pp_bandwidth_bytes_s() * opt.continuous_batching_efficiency
            if bw > 0:
                comm.pp_time_per_token_ms = (comm.pp_bytes_per_token / bw) * 1000.0

        if self.cp > 1:
            kv_bytes_per_token = self.kv_bytes_per_token * opt.kv_compression_ratio
            # Ring all-gather: each GPU holds 1/cp of KV; total data moved is
            # (cp-1)/cp of the full KV cache.  kv_bytes_per_token already spans
            # all layers, so do not multiply by n_layers again.
            comm.cp_bytes_per_token = kv_bytes_per_token * (self.cp - 1) / self.cp
            bw = self._cp_bandwidth_bytes_s() * opt.continuous_batching_efficiency
            if bw > 0:
                comm.cp_time_per_token_ms = (comm.cp_bytes_per_token / bw) * 1000.0

        if phase == "decode":
            comm.kv_offload_time_ms = self.kv_offload_time_per_token * 1000.0
            comm.afd_transfer_time_ms = self.afd_transfer_time_per_token * 1000.0

        return comm

    def _infer_bottleneck(
        self,
        prefill_breakdown: Dict[str, float],
        decode_breakdown: Dict[str, float],
        feasible: bool,
    ) -> str:
        """Classify the dominant inference bottleneck from latency breakdowns."""
        if not feasible:
            return "memory: HBM overflow"

        prefill_total = sum(prefill_breakdown.values())
        decode_total = sum(decode_breakdown.values())
        # We care about the steady-state bottleneck; decode usually dominates
        # serving latency/throughput.
        if decode_total >= prefill_total:
            breakdown = decode_breakdown
            phase = "decode"
        else:
            breakdown = prefill_breakdown
            phase = "prefill"

        # Drop components that are not the dominant path for this phase.
        # For decode, compute/memory are alternatives (max); for prefill,
        # MLP compute and attention are sequential.
        if phase == "prefill":
            dominant_compute = breakdown.get("compute", 0.0) + breakdown.get("attention", 0.0)
        else:
            dominant_compute = max(breakdown.get("compute", 0.0), breakdown.get("memory", 0.0))
        comm = breakdown.get("communication", 0.0)
        offload = breakdown.get("kv_offload", 0.0) + breakdown.get("afd_transfer", 0.0)
        overhead = breakdown.get("overhead", 0.0)

        if comm > dominant_compute and comm > offload and comm > overhead:
            return f"network: {phase} communication bottleneck"
        if offload > dominant_compute and offload > overhead:
            return f"memory: {phase} offload/transfer bottleneck"
        if overhead > dominant_compute and overhead > comm:
            return f"software: {phase} scheduler/overhead bottleneck"
        if phase == "decode" and breakdown.get("memory", 0.0) > breakdown.get("compute", 0.0):
            return f"memory: {phase} bandwidth bottleneck"
        if phase == "prefill" and breakdown.get("attention", 0.0) > breakdown.get("compute", 0.0):
            return f"compute: {phase} attention-bound"
        return f"compute: {phase} GPU-bound"

    def _network_utilization(
        self,
        total_tokens: int,
        duration: float,
        comm: CommunicationBreakdown,
    ) -> float:
        """Estimate peak inter-node network bandwidth utilization (0-1)."""
        if duration <= 0 or total_tokens <= 0:
            return 0.0

        total_comm_bytes_per_token = (
            comm.tp_bytes_per_token + comm.pp_bytes_per_token + comm.cp_bytes_per_token
        )
        comm_bytes_per_sec = (total_tokens / duration) * total_comm_bytes_per_token

        if self.cluster is not None:
            total_inter_node_bw = (
                self.cluster.topology.aggregate_inter_node_bw_gb_s
                * self.cluster.node_count
                * 1e9
            )
        else:
            # Fallback: treat each GPU as having its own interconnect link.
            total_inter_node_bw = self.hardware.interconnect_bw_gb_s * self.gpu_count * 1e9

        if total_inter_node_bw <= 0:
            return 0.0
        return min(1.0, comm_bytes_per_sec / total_inter_node_bw)

    def _prefill_utilization(self, seq_len: int) -> float:
        """Sequence-length-aware prefill utilization.

        Small prefill kernels (short context, small batch) suffer from CUDA
        launch overhead, kernel scheduling gaps, and poor GEMM efficiency.
        Utilization rises with sequence length and approaches the configured
        ceiling for long-context prefill.  This is the structural fix behind
        Path 2: a single global constant cannot fit both 2K and 131K contexts.
        """
        max_util = sim_config.DEFAULT_PREFILL_UTILIZATION
        saturation = sim_config.DEFAULT_PREFILL_SATURATION_TOKENS
        if saturation <= 0:
            return max_util
        return max_util * seq_len / (seq_len + saturation)

    def _compute_prefill_time_per_token(self) -> float:
        """Time to process one token during prefill (compute-bound + comm)."""
        breakdown = self._compute_prefill_breakdown()
        return sum(breakdown.values())

    def _attention_time_per_token(self, seq_len: int) -> float:
        """Memory/compute roofline estimate for FlashAttention-style prefill.

        Models attention as either HBM-bandwidth-bound (short context) or
        FLOP-bound (very long context).  The "passes" constant captures how
        many times the per-layer activation is read/written from HBM; it is
        calibrated against real serving fixtures rather than guessed.

        CP parallelism reduces the effective sequence length seen by each
        GPU; compressed KV families (MLA) reduce per-token HBM traffic.
        """
        passes = sim_config.DEFAULT_PREFILL_ATTENTION_HBM_PASSES
        if passes <= 0:
            return 0.0

        n_layers = self.model.n_layers
        d_model = self.model.d_model
        num_gpus = self._prefill_gpu_count()
        util = self._prefill_utilization(seq_len)

        # Sequence parallelism: attention work is divided across CP group.
        effective_seq_len = seq_len / max(1, self.cp)

        # KV compression (MLA) reduces attention memory traffic.
        kv_scale = 1.0
        family = self.model.algorithm_family
        if family and family.kv_scaling == "compressed":
            kv_scale = max(0.01, family.default_kv_compression_ratio)

        # FlashAttention-style kernels become more memory-efficient as the
        # sequence length grows because each tile reuses more data.  Model this
        # by decaying the effective HBM passes from the calibrated ceiling down
        # to a physical floor of ~4 passes (Q, K, V, O) per layer per token.
        min_passes = 4.0
        saturation = sim_config.DEFAULT_PREFILL_SATURATION_TOKENS
        if saturation > 0:
            effective_passes = min_passes + (passes - min_passes) * saturation / (seq_len + saturation)
        else:
            effective_passes = passes

        # Memory-bound path: each effective pass reads/writes one activation
        # vector per layer per token.
        hbm_bytes_per_token = (
            effective_passes * n_layers * d_model * self.bytes_per_param * kv_scale
        )
        mem_time = hbm_bytes_per_token / (
            self.hardware.memory_bw_bytes_s()
            * max(1, num_gpus)
            * util
        )

        # Compute-bound path: O(seq_len * d_model * n_layers) FLOPs per token.
        flops_per_token = 2 * n_layers * d_model * effective_seq_len
        eff_flops = self.hardware.effective_flops(self.precision, util) * max(1, num_gpus)
        compute_time = flops_per_token / eff_flops

        return max(mem_time, compute_time)

    def _compute_prefill_breakdown(self) -> Dict[str, float]:
        """Breakdown of prefill time per token into components (seconds)."""
        seq_len = max(1, self.service_config.request_length_mean)
        util = self._prefill_utilization(seq_len)
        flops_per_token = self.model.flops_per_token_forward()
        effective_flops = (
            self.hardware.effective_flops(self.precision, util)
            * self._prefill_gpu_count()
        )
        compute_time = flops_per_token / effective_flops

        comm = self._communication_breakdown(seq_len, 1, phase="prefill")
        comm_time = (
            comm.tp_time_per_token_ms + comm.pp_time_per_token_ms + comm.cp_time_per_token_ms
        ) / 1000.0

        # Per-request system latency floor: scheduler, kernel launch, CUDA
        # graph setup, PagedAttention block allocation.  Spread over the input
        # sequence so the total contribution is constant per request.
        floor_s = sim_config.DEFAULT_PREFILL_LATENCY_FLOOR_MS / 1000.0
        floor_per_token = floor_s / seq_len

        return {
            "compute": compute_time,
            "attention": self._attention_time_per_token(seq_len),
            "communication": comm_time,
            "floor": floor_per_token,
            "overhead": self.optimization.prefill_overhead_ms / 1000.0,
        }

    def _compute_decode_time_per_token(self) -> float:
        """Time to generate one token during decode (memory/compute-bound + comm + offload)."""
        breakdown = self._compute_decode_breakdown()
        return sum(breakdown.values())

    def _compute_decode_breakdown(self) -> Dict[str, float]:
        """Breakdown of decode time per token into components (seconds)."""
        flops_per_token = self.model.flops_per_token_forward()
        effective_compute = (
            self.hardware.effective_flops(self.precision, sim_config.DEFAULT_DECODE_UTILIZATION)
            * self._decode_gpu_count()
        )
        compute_time = flops_per_token / effective_compute

        active_params = self.model.active_params_b * 1e9
        weight_bytes = active_params * self.bytes_per_param
        kv_bytes = self.kv_bytes_per_token
        effective_bw = (
            self.hardware.memory_bw_bytes_s()
            * sim_config.DEFAULT_DECODE_UTILIZATION
            * self._decode_gpu_count()
        )
        effective_decode_batch = max(1.0, self.service_config.max_batch_size / 2.0)
        memory_time = weight_bytes / (effective_bw * effective_decode_batch) + kv_bytes / effective_bw

        comm = self._communication_breakdown(1, 1, phase="decode")
        comm_time = (
            comm.tp_time_per_token_ms + comm.pp_time_per_token_ms + comm.cp_time_per_token_ms
        ) / 1000.0

        offload_time = self.kv_offload_time_per_token
        afd_time = self.afd_transfer_time_per_token

        decode_overhead_s = self.optimization.decode_overhead_ms / 1000.0
        if decode_overhead_s == 0:
            # No fixture override: apply architecture-specific default overhead.
            decode_overhead_s = (10.0 if self.model.is_moe else 4.0) / 1000.0
        elif decode_overhead_s < 0:
            # Negative value is an explicit "disable default overhead" signal.
            decode_overhead_s = 0.0

        return {
            "compute": compute_time,
            "memory": memory_time,
            "communication": comm_time,
            "kv_offload": offload_time,
            "afd_transfer": afd_time,
            "overhead": decode_overhead_s,
        }

    def _effective_transfer_bw_gb_s(self, requested_bw_gb_s: float) -> float:
        """Cap requested pool-to-pool bandwidth by cluster inter-node bandwidth."""
        if self.cluster is None:
            return requested_bw_gb_s
        cluster_bw = self.cluster.topology.effective_inter_node_bw_gb_s
        if requested_bw_gb_s <= 0:
            return cluster_bw
        return min(requested_bw_gb_s, cluster_bw)

    def _compute_kv_transfer_time_per_request(self) -> float:
        """Time to transfer KV cache from prefill pool to decode pool (PD)."""
        if not self.pd_config.enabled:
            return 0.0
        cfg = self.service_config
        avg_context_tokens = cfg.request_length_mean
        kv_bytes = avg_context_tokens * self.kv_bytes_per_token * self.optimization.kv_compression_ratio
        bw_gb_s = self._effective_transfer_bw_gb_s(self.pd_config.kv_transfer_bw_gb_s)
        bw_bytes_s = bw_gb_s * 1e9
        if bw_bytes_s <= 0:
            return 0.0
        transfer_time = kv_bytes / bw_bytes_s + self.pd_config.kv_transfer_latency_us * 1e-6
        # Chunking overhead: number of chunks * latency
        chunk_size_bytes = self.pd_config.transfer_chunk_size_mb * 1e6
        if chunk_size_bytes > 0:
            n_chunks = max(1, int(np.ceil(kv_bytes / chunk_size_bytes)))
            transfer_time += (n_chunks - 1) * self.pd_config.kv_transfer_latency_us * 1e-6
        return transfer_time

    def _compute_kv_offload_time_per_token(self) -> float:
        """Expected KV cache access time per decode token due to tiered offload."""
        return self.optimization.effective_kv_access_time_s(self.kv_bytes_per_token)

    def _compute_afd_transfer_time_per_token(self) -> float:
        """Activation transfer time for AFD-separated prefill/decode."""
        if not self.afd_config.enabled:
            return 0.0
        # In a pipelined AFD deployment, only one layer's activation is in
        # flight across the attention-FFN boundary at a time.  The per-token
        # transfer footprint is therefore 2 * d_model (attention output -> FFN
        # and FFN output -> next attention), not 2 * n_layers * d_model.
        hidden_bytes = self.model.d_model * self.bytes_per_param
        bytes_per_token = 2 * hidden_bytes
        bw_gb_s = self._effective_transfer_bw_gb_s(self.afd_config.activation_transfer_bw_gb_s)
        bw_bytes_s = bw_gb_s * 1e9
        if bw_bytes_s <= 0:
            return 0.0
        return bytes_per_token / bw_bytes_s + self.afd_config.activation_transfer_latency_us * 1e-6

    def _generate_requests(self) -> List[Request]:
        """Generate a sequence of requests based on arrival distribution."""
        cfg = self.service_config
        duration = cfg.simulation_duration_seconds

        if cfg.arrival_distribution == "poisson":
            n_requests = int(cfg.arrival_rate_per_sec * duration * 2)
            inter_arrivals = self.rng.exponential(1.0 / cfg.arrival_rate_per_sec, size=n_requests)
            arrival_times = np.cumsum(inter_arrivals)
            arrival_times = arrival_times[arrival_times <= duration]
        elif cfg.arrival_distribution == "fixed":
            n_requests = int(cfg.arrival_rate_per_sec * duration)
            arrival_times = np.linspace(0, duration, n_requests, endpoint=False)
        else:
            n_requests = int(cfg.arrival_rate_per_sec * duration)
            inter_arrivals = self.rng.exponential(1.0 / cfg.arrival_rate_per_sec, size=n_requests)
            arrival_times = np.cumsum(inter_arrivals)
            arrival_times = arrival_times[arrival_times <= duration]

        requests = []
        for i, t in enumerate(arrival_times):
            input_len = max(1, int(self.rng.normal(cfg.request_length_mean, cfg.request_length_std)))
            output_len = max(1, int(self.rng.normal(cfg.output_length_mean, cfg.output_length_std)))
            requests.append(Request(id=i, arrival_time=t, input_len=input_len, output_len=output_len))
        return requests

    def _memory_for_batch(self, requests: List[Request]) -> float:
        """Estimate memory (GB) required for a set of active requests per GPU."""
        total_tokens = sum(r.input_len + r.generated_tokens for r in requests)
        return self._memory_per_gpu_gb(total_tokens, batch_size=1)

    def _next_arrival_time(self, arrivals: List[Request], idx: int) -> float:
        if idx < len(arrivals):
            return arrivals[idx].arrival_time
        return float("inf")

    def _next_prefill_completion(self, prefill_batch: List[Request]) -> float:
        if not prefill_batch:
            return float("inf")
        total_tokens = sum(r.input_len for r in prefill_batch)
        return total_tokens * self.prefill_time_per_token

    def _next_decode_completion(self, decode_batch: List[Request]) -> float:
        if not decode_batch:
            return float("inf")
        # In continuous batching all tokens in the batch are generated in parallel,
        # so the time to complete one decode step is independent of batch size.
        return self.decode_time_per_token

    def _apply_residual_correction(
        self, analytical_prefill_ms: float, analytical_tpot_ms: float
    ) -> Dict[str, float]:
        """Apply trained residual correction to analytical latency estimates.

        Loads the default serving residual model if present and adjusts TTFT/TPOT
        estimates based on workload features.  Falls back to baseline estimates
        when no model is available.
        """
        default_path = Path(__file__).parent.parent / "calibration" / "data" / "residual_model_serving.json"
        if not default_path.exists():
            return {"ttft_p50_ms": analytical_prefill_ms, "tpot_p50_ms": analytical_tpot_ms}

        try:
            residual_model = ResidualModel.load(str(default_path))
        except Exception:
            return {"ttft_p50_ms": analytical_prefill_ms, "tpot_p50_ms": analytical_tpot_ms}

        seq_len = max(1, self.service_config.request_length_mean)
        features = {
            "seq_len": seq_len,
            "model_params_b": float(self.model.total_params_b),
            "gpu_count": int(self.gpu_count),
            "is_pd": bool(self.pd_config.enabled),
            "is_afd": bool(self.afd_config.enabled),
            "is_moe": bool(self.model.is_moe),
            "cp": int(self.cp),
        }
        predicted = {
            "ttft_p50_ms": analytical_prefill_ms,
            "tpot_p50_ms": analytical_tpot_ms,
        }
        corrected = residual_model.apply("serve", features, predicted)
        return corrected

    def run(self) -> SimulationResult:
        """Run event-driven inference serving simulation with pool-level parallelism."""
        cfg = self.service_config
        arrivals = self._generate_requests()
        arrival_idx = 0

        queue: List[Request] = []
        pending_decode: List[Request] = []
        completed: List[Request] = []
        dropped = 0

        # Independent slots model data-parallel instances inside prefill/decode pools.
        prefill_slots: List[List[Request]] = [[] for _ in range(self.prefill_instances)]
        prefill_remaining: List[float] = [0.0] * self.prefill_instances
        decode_slots: List[List[Request]] = [[] for _ in range(self.decode_instances)]
        decode_remaining: List[float] = [0.0] * self.decode_instances

        current_time = 0.0
        max_queue_len = cfg.max_queue_len
        max_batch_size = cfg.max_batch_size

        # Feasibility check for distributed memory
        max_context_tokens = cfg.request_length_mean + cfg.output_length_mean
        feasible = self._fits_in_cluster(max_context_tokens, batch_size=max_batch_size)

        def _fill_prefill_slot(slot_idx: int) -> None:
            """Fill an empty prefill slot from the queue until memory or batch limit."""
            if prefill_slots[slot_idx]:
                return
            batch = []
            while queue and len(batch) < max_batch_size:
                candidate = queue[0]
                test_batch = batch + [candidate]
                total_tokens = sum(r.input_len for r in test_batch)
                if (
                    self._memory_per_gpu_gb(total_tokens, batch_size=1)
                    <= self.hardware.memory_gb * FEASIBILITY_MAX_MEMORY_UTIL
                ):
                    req = queue.pop(0)
                    req.start_prefill_time = current_time
                    batch.append(req)
                else:
                    break
            if batch:
                prefill_slots[slot_idx] = batch
                prefill_remaining[slot_idx] = self._next_prefill_completion(batch)

        def _fill_decode_slot(slot_idx: int) -> None:
            """Fill or top-up a decode slot from pending decode requests."""
            batch = decode_slots[slot_idx]
            while pending_decode and len(batch) < max_batch_size:
                candidate = pending_decode[0]
                test_batch = batch + [candidate]
                total_tokens = sum(r.input_len + r.generated_tokens for r in test_batch)
                if (
                    self._memory_per_gpu_gb(total_tokens, batch_size=1)
                    <= self.hardware.memory_gb * FEASIBILITY_MAX_MEMORY_UTIL
                ):
                    batch.append(pending_decode.pop(0))
                else:
                    break
            if batch and not decode_slots[slot_idx]:
                decode_slots[slot_idx] = batch
                decode_remaining[slot_idx] = self._next_decode_completion(batch)
            elif batch and decode_slots[slot_idx] and decode_remaining[slot_idx] <= 1e-9:
                # Continuous batching: slot was just finishing a step; re-arm.
                decode_remaining[slot_idx] = self._next_decode_completion(batch)

        while (
            arrival_idx < len(arrivals)
            or queue
            or pending_decode
            or any(prefill_slots)
            or any(decode_slots)
        ):
            # Admit new arrivals
            while arrival_idx < len(arrivals) and arrivals[arrival_idx].arrival_time <= current_time:
                req = arrivals[arrival_idx]
                if len(queue) >= max_queue_len:
                    req.dropped = True
                    dropped += 1
                else:
                    queue.append(req)
                arrival_idx += 1

            # Fill empty slots
            for i in range(self.prefill_instances):
                _fill_prefill_slot(i)
            for i in range(self.decode_instances):
                _fill_decode_slot(i)

            # Determine next event time
            next_arrival = self._next_arrival_time(arrivals, arrival_idx)
            next_prefill = min(
                (t for t, slot in zip(prefill_remaining, prefill_slots) if slot),
                default=float("inf"),
            )
            next_decode = min(
                (t for t, slot in zip(decode_remaining, decode_slots) if slot),
                default=float("inf"),
            )

            next_event_time = min(next_arrival, next_prefill, next_decode)
            if next_event_time == float("inf"):
                break

            # Advance time
            advance = next_event_time
            current_time += advance

            for i in range(self.prefill_instances):
                if prefill_slots[i]:
                    prefill_remaining[i] -= advance
            for i in range(self.decode_instances):
                if decode_slots[i]:
                    decode_remaining[i] -= advance

            # Handle decode completions
            for i in range(self.decode_instances):
                if decode_slots[i] and decode_remaining[i] <= 1e-9:
                    for req in decode_slots[i]:
                        req.generated_tokens += 1
                    still_decoding = []
                    for req in decode_slots[i]:
                        if req.generated_tokens >= req.output_len:
                            req.completion_time = current_time
                            completed.append(req)
                        else:
                            still_decoding.append(req)
                    decode_slots[i] = still_decoding
                    decode_remaining[i] = 0.0
                    # Continuous batching: backfill finished requests immediately
                    _fill_decode_slot(i)

            # Handle prefill completions
            for i in range(self.prefill_instances):
                if prefill_slots[i] and prefill_remaining[i] <= 1e-9:
                    transfer_complete_time = current_time + self.kv_transfer_time_per_request
                    for req in prefill_slots[i]:
                        req.first_token_time = transfer_complete_time
                        req.kv_transferred = True
                        pending_decode.append(req)
                    prefill_slots[i] = []
                    prefill_remaining[i] = 0.0

        # Compute metrics
        completed_reqs = [r for r in completed if not r.dropped]
        ttfts = [r.ttft for r in completed_reqs if r.ttft >= 0]
        tpots = [
            (r.total_latency - r.ttft) / r.output_len
            for r in completed_reqs
            if r.total_latency >= 0 and r.output_len > 0
        ]
        total_latencies = [r.total_latency for r in completed_reqs if r.total_latency >= 0]

        total_tokens = sum(r.input_len + r.output_len for r in completed_reqs)
        duration = cfg.simulation_duration_seconds

        throughput_req = len(completed_reqs) / duration if duration > 0 else 0.0
        throughput_tok = total_tokens / duration if duration > 0 else 0.0

        total_decode_tokens = sum(r.output_len for r in completed_reqs)
        busy_time = (
            sum(r.input_len for r in completed_reqs) * self.prefill_time_per_token
            + total_decode_tokens * self.decode_time_per_token
        )
        utilization = min(1.0, busy_time / (duration * self.gpu_count)) if duration > 0 else 0.0

        cost_usd = (duration / 3600.0) * self.hardware.cost_per_hour * self.gpu_count

        sim_cfg = SimulationConfig(
            hardware=self.hardware,
            model=self.model,
            workload=WorkloadSpec(
                name="inference-serving",
                max_steps=1,
                avg_steps=1.0,
                step_std=0.0,
                context_limit=cfg.request_length_mean * 2,
            ),
            harness=AgentHarnessSpec(name="inference-serving", concurrency=1),
            target_context_tokens=cfg.request_length_mean,
            precision=self.precision,
            kv_precision=self.kv_precision,
            optimization=self.optimization,
        )

        def _p(values, p):
            if not values:
                return 0.0
            return float(np.percentile(values, p))

        mean_comm = self._communication_breakdown(1, 1, phase="decode")
        prefill_breakdown = self._compute_prefill_breakdown()
        decode_breakdown = self._compute_decode_breakdown()
        bottleneck = self._infer_bottleneck(prefill_breakdown, decode_breakdown, feasible)
        network_util = self._network_utilization(total_tokens, duration, mean_comm)

        # Use analytical latency estimates for TTFT/TPOT p50; p99 is derived
        # from p50.  Queueing is intentionally not modeled here because the
        # batch-formation latency is small relative to compute/transfer for the
        # calibrated fixtures; throughput and drops are captured by the event
        # loop.  Event-driven percentiles are retained in metadata for
        # diagnostics.
        analytical_prefill_ms = (
            self.prefill_time_per_token * cfg.request_length_mean * 1000.0
            + self.kv_transfer_time_per_request * 1000.0
            + self.afd_transfer_time_per_token * cfg.request_length_mean * 1000.0
        )
        analytical_tpot_ms = self.decode_time_per_token * 1000.0

        # Apply trained residual correction if a default residual model exists.
        # This is the hybrid analytical + ML step that brings TTFT/TPOT errors
        # down to the calibrated range.
        corrected = self._apply_residual_correction(
            analytical_prefill_ms, analytical_tpot_ms
        )
        analytical_prefill_ms = corrected.get("ttft_p50_ms", analytical_prefill_ms)
        analytical_tpot_ms = corrected.get("tpot_p50_ms", analytical_tpot_ms)

        return SimulationResult(
            config=sim_cfg,
            latency_seconds=duration,
            wall_time_seconds=duration,
            tokens_total=total_tokens,
            tokens_input=sum(r.input_len for r in completed_reqs),
            tokens_output=sum(r.output_len for r in completed_reqs),
            peak_kv_gb=max(
                (self._memory_per_gpu_gb(cfg.request_length_mean + cfg.output_length_mean, 1)
                 - self.weight_memory_gb / max(1, self.tp * self.pp)),
                0.0,
            ) * self.gpu_count / max(1, self.cp),
            memory_required_gb=self.hardware.memory_gb * self.gpu_count,
            gpu_count=self.gpu_count,
            feasible=feasible,
            bottleneck=bottleneck,
            cost_usd=cost_usd,
            utilization_gpu=utilization,
            metadata={
                "requests_total": len(arrivals),
                "requests_completed": len(completed_reqs),
                "requests_dropped": dropped,
                "throughput_req_per_sec": throughput_req,
                "throughput_tok_per_sec": throughput_tok,
                "ttft_p50_ms": analytical_prefill_ms,
                "ttft_p99_ms": analytical_prefill_ms * 2.0,
                "tpot_p50_ms": analytical_tpot_ms,
                "tpot_p99_ms": analytical_tpot_ms * 2.0,
                "event_ttft_p50_ms": _p(ttfts, 50) * 1000.0 if ttfts else 0.0,
                "event_ttft_p99_ms": _p(ttfts, 99) * 1000.0 if ttfts else 0.0,
                "event_tpot_p50_ms": _p(tpots, 50) * 1000.0 if tpots else 0.0,
                "event_tpot_p99_ms": _p(tpots, 99) * 1000.0 if tpots else 0.0,
                "e2e_latency_p99_ms": _p(total_latencies, 99) * 1000,
                "prefill_time_per_token_ms": self.prefill_time_per_token * 1000,
                "decode_time_per_token_ms": self.decode_time_per_token * 1000,
                "kv_transfer_time_per_request_ms": self.kv_transfer_time_per_request * 1000,
                "kv_offload_time_per_token_ms": self.kv_offload_time_per_token * 1000,
                "afd_transfer_time_per_token_ms": self.afd_transfer_time_per_token * 1000,
                "prefill_breakdown_ms": {k: v * 1000.0 for k, v in prefill_breakdown.items()},
                "decode_breakdown_ms": {k: v * 1000.0 for k, v in decode_breakdown.items()},
                "network_utilization": network_util,
                "tp": self.tp,
                "pp": self.pp,
                "cp": self.cp,
                "pd_enabled": self.pd_config.enabled,
                "prefill_gpu_count": self._prefill_gpu_count(),
                "decode_gpu_count": self._decode_gpu_count(),
                "afd_enabled": self.afd_config.enabled,
                "memory_per_gpu_gb": self._memory_per_gpu_gb(
                    cfg.request_length_mean + cfg.output_length_mean, batch_size=1
                ),
                "tp_comm_bytes_per_token": mean_comm.tp_bytes_per_token,
                "pp_comm_bytes_per_token": mean_comm.pp_bytes_per_token,
                "cp_comm_bytes_per_token": mean_comm.cp_bytes_per_token,
                "tp_comm_time_ms": mean_comm.tp_time_per_token_ms,
                "pp_comm_time_ms": mean_comm.pp_time_per_token_ms,
                "cp_comm_time_ms": mean_comm.cp_time_per_token_ms,
            },
        )


def run_serving(
    model: ModelSpec,
    hardware: HardwareSpec,
    service_config: InferenceServiceConfig,
    precision: str = "FP8",
    kv_precision: str = "FP8",
    optimization: OptimizationConfig = None,
    gpu_count: int = 1,
    cluster: ClusterSpec = None,
    tp: int = 1,
    pp: int = 1,
    cp: int = 1,
    seed: int = 42,
) -> SimulationResult:
    """Convenience function to run inference serving simulation."""
    return InferenceServingEngine(
        model=model,
        hardware=hardware,
        service_config=service_config,
        precision=precision,
        kv_precision=kv_precision,
        optimization=optimization,
        gpu_count=gpu_count,
        cluster=cluster,
        tp=tp,
        pp=pp,
        cp=cp,
        seed=seed,
    ).run()
