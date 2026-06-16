"""Core simulation engine for agent workload × hardware co-design."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..config import DEFAULT_PREFILL_UTILIZATION, DEFAULT_DECODE_UTILIZATION, FEASIBILITY_MEMORY_OVERHEAD, FEASIBILITY_MAX_MEMORY_UTIL
from ..data_models import (
    AgentHarnessSpec,
    HardwareSpec,
    ModelSpec,
    OptimizationConfig,
    SimulationConfig,
    SimulationResult,
    WorkloadSpec,
)
from ..utils.stats import sample_discrete
from ..utils.units import bytes_to_gb, gb_to_bytes


@dataclass
class StepMetrics:
    """Metrics for a single ReAct step."""

    step: int
    tool: str
    thought_tokens: int
    action_tokens: int
    obs_tokens: int
    prefill_input: int
    decode_output: int
    prefill_time_ms: float
    decode_time_ms: float
    tool_time_ms: float
    hbf_penalty_ms: float
    kv_total_gb: float
    kv_hbm_gb: float
    kv_hbf_gb: float
    history_tokens: int
    truncated: int


class SimulationEngine:
    """Simulate a single agent task end-to-end.

    Combines a model, hardware, workload, harness, and optimization stack to
    produce latency, token, memory, and cost estimates.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self._compute_derived()

    def _compute_derived(self) -> None:
        """Pre-compute deployment and throughput numbers."""
        cfg = self.config
        hw = cfg.hardware
        model = cfg.model
        opt = cfg.optimization

        precision = cfg.precision.upper()
        kv_precision = cfg.kv_precision.upper()
        bytes_per_param = model.bytes_per_param(precision)
        self.kv_bytes_per_token = model.kv_bytes_per_token(kv_precision) * opt.kv_compression_ratio

        # Weight memory on a single device
        self.weight_memory_gb = model.weight_memory_gb(precision)

        # Initial GPU count estimate based on weights alone
        base_gpus = model.gpu_needed(hw.memory_gb, precision, overhead=FEASIBILITY_MEMORY_OVERHEAD)

        # Adjust GPU count if target context KV does not fit
        kv_for_context_gb = (
            cfg.target_context_tokens * self.kv_bytes_per_token / gb_to_bytes(1.0)
        )
        total_memory_gb = self.weight_memory_gb * FEASIBILITY_MEMORY_OVERHEAD + kv_for_context_gb
        required_gpus_by_memory = int(
            np.ceil(total_memory_gb / (hw.memory_gb * FEASIBILITY_MAX_MEMORY_UTIL))
        )
        self.gpu_count = max(base_gpus, required_gpus_by_memory)

        # Throughput estimates
        active_params = model.active_params_b * 1e9
        flops_per_token = 2 * active_params

        # Prefill: process the whole prompt at once
        effective_prefill_flops = (
            hw.effective_flops(precision, DEFAULT_PREFILL_UTILIZATION)
            * opt.effective_prefill_speedup()
            * self.gpu_count
        )
        self.prefill_time_per_token = flops_per_token / effective_prefill_flops

        # Decode: memory-bound or compute-bound, take the slower
        effective_decode_compute_flops = (
            hw.effective_flops(precision, DEFAULT_DECODE_UTILIZATION)
            * opt.effective_decode_speedup()
            * self.gpu_count
        )
        decode_compute_time = flops_per_token / effective_decode_compute_flops

        # Memory traffic: active weights + KV read per generated token
        bytes_per_decode_token = (
            active_params * bytes_per_param + self.kv_bytes_per_token
        )
        effective_bw = hw.memory_bw_bytes_s() * DEFAULT_DECODE_UTILIZATION * self.gpu_count
        decode_memory_time = bytes_per_decode_token / effective_bw

        self.decode_time_per_token = max(decode_compute_time, decode_memory_time)

        # HBF offload penalty per decode token (simplified)
        self.hbf_penalty_per_token = 0.0
        if opt.hbf_offload and opt.hbf_offload_ratio > 0:
            hbf_bytes_per_token = self.kv_bytes_per_token * opt.hbf_offload_ratio
            self.hbf_penalty_per_token = hbf_bytes_per_token / gb_to_bytes(opt.hbf_bw_gb_s)

    def _sample_steps(self) -> int:
        """Sample number of ReAct steps for this task."""
        workload = self.config.workload
        steps = int(self.rng.normal(workload.avg_steps, workload.step_std))
        return max(5, min(steps, workload.max_steps))

    def _sample_tool(self, is_final: bool) -> Tuple[str, float]:
        """Sample tool type and delay."""
        workload = self.config.workload
        if is_final:
            return "submit", workload.tool_delays.get("submit", 0.01)
        tool = sample_discrete(self.rng, workload.tool_probs)
        delay = workload.tool_delays.get(tool, 0.1)
        return tool, delay

    def _sample_tokens(self) -> Tuple[int, int, int]:
        """Sample (thought, action, observation) token counts."""
        dists = self.config.workload.token_distributions
        thought = max(50, int(self.rng.normal(*dists["thought"])))
        action = max(20, int(self.rng.normal(*dists["action"])))
        obs = max(30, int(self.rng.normal(*dists["observation"])))
        return thought, action, obs

    def run(self) -> SimulationResult:
        """Run a single end-to-end simulation."""
        cfg = self.config
        workload = cfg.workload
        opt = cfg.optimization

        n_steps = self._sample_steps()
        system_prompt_tokens = 2000
        issue_desc_tokens = 1500
        history_tokens = system_prompt_tokens + issue_desc_tokens

        steps_data: List[Dict] = []
        total_prefill_tokens = 0
        total_decode_tokens = 0
        total_llm_time = 0.0
        total_tool_time = 0.0
        kv_cache_peak_gb = 0.0
        kv_hbm_peak_gb = 0.0
        kv_hbf_peak_gb = 0.0

        for step_idx in range(n_steps):
            is_final = step_idx == n_steps - 1
            tool, tool_delay = self._sample_tool(is_final)
            thought_tok, action_tok, obs_tok = self._sample_tokens()

            decode_output = thought_tok + action_tok

            # Prefill input: with prefix caching only new tokens need full prefill
            if opt.prefix_caching and step_idx > 0:
                new_input = obs_tok + 50
                cache_hit_tokens = history_tokens - new_input
                prefill_input = int(
                    new_input + (1 - opt.prefix_caching_hit_rate) * max(0, cache_hit_tokens)
                )
            else:
                prefill_input = history_tokens

            prefill_time = prefill_input * self.prefill_time_per_token

            # Decode: generate thought + action
            decode_time = decode_output * self.decode_time_per_token

            # HBF penalty applies to the fraction of KV kept in HBF
            hbf_penalty = decode_output * self.hbf_penalty_per_token

            # Tool execution
            tool_time = tool_delay

            # KV cache accounting
            if opt.prefix_caching and step_idx > 0:
                actual_new_kv_tokens = decode_output + int(
                    (1 - opt.prefix_caching_hit_rate) * prefill_input
                )
            else:
                actual_new_kv_tokens = prefill_input + decode_output

            total_kv_tokens = history_tokens + decode_output
            kv_total_gb = total_kv_tokens * self.kv_bytes_per_token / gb_to_bytes(1.0)

            # HBM/HBF split
            if opt.hbf_offload:
                hbm_available_for_kv = max(
                    0.0,
                    (cfg.hardware.memory_gb * self.gpu_count / FEASIBILITY_MEMORY_OVERHEAD)
                    - self.weight_memory_gb,
                )
                # Reserve some HBM headroom
                hbm_kv_capacity = hbm_available_for_kv * (1 - opt.hbf_offload_ratio)
                hbm_kv_gb = min(kv_total_gb, hbm_kv_capacity)
                hbf_kv_gb = max(0.0, kv_total_gb - hbm_kv_gb)
            else:
                hbm_kv_gb = kv_total_gb
                hbf_kv_gb = 0.0

            kv_cache_peak_gb = max(kv_cache_peak_gb, kv_total_gb)
            kv_hbm_peak_gb = max(kv_hbm_peak_gb, hbm_kv_gb)
            kv_hbf_peak_gb = max(kv_hbf_peak_gb, hbf_kv_gb)

            # Update history
            history_tokens += decode_output + obs_tok
            truncated = 0
            if history_tokens > workload.context_limit:
                truncated = history_tokens - workload.context_limit
                history_tokens = workload.context_limit

            step = StepMetrics(
                step=step_idx + 1,
                tool=tool,
                thought_tokens=thought_tok,
                action_tokens=action_tok,
                obs_tokens=obs_tok,
                prefill_input=prefill_input,
                decode_output=decode_output,
                prefill_time_ms=prefill_time * 1000,
                decode_time_ms=decode_time * 1000,
                tool_time_ms=tool_time * 1000,
                hbf_penalty_ms=hbf_penalty * 1000,
                kv_total_gb=kv_total_gb,
                kv_hbm_gb=hbm_kv_gb,
                kv_hbf_gb=hbf_kv_gb,
                history_tokens=history_tokens,
                truncated=truncated,
            )
            steps_data.append(step.__dict__)

            total_prefill_tokens += prefill_input
            total_decode_tokens += decode_output
            total_llm_time += prefill_time + decode_time + hbf_penalty
            total_tool_time += tool_time

        env_init_time = workload.env_init_delay
        total_time = env_init_time + total_llm_time + total_tool_time

        # Memory required on the whole allocation
        memory_required_gb = (
            self.weight_memory_gb + kv_cache_peak_gb
        ) * FEASIBILITY_MEMORY_OVERHEAD

        # Feasibility: fits in allocated GPUs
        available_memory_gb = cfg.hardware.memory_gb * self.gpu_count
        feasible = memory_required_gb <= available_memory_gb * FEASIBILITY_MAX_MEMORY_UTIL

        # Bottleneck identification
        bottleneck = self._identify_bottleneck(
            total_llm_time, total_tool_time, total_time, feasible, memory_required_gb, available_memory_gb
        )

        # Cost: GPU hours
        cost_usd = (total_time / 3600.0) * cfg.hardware.cost_per_hour * self.gpu_count

        # GPU utilization (time-weighted approximation)
        utilization_gpu = total_llm_time / total_time if total_time > 0 else 0.0

        result = SimulationResult(
            config=cfg,
            latency_seconds=total_time,
            wall_time_seconds=total_time / max(1, cfg.harness.concurrency),
            tokens_total=total_prefill_tokens + total_decode_tokens,
            tokens_input=total_prefill_tokens,
            tokens_output=total_decode_tokens,
            peak_kv_gb=kv_cache_peak_gb,
            memory_required_gb=memory_required_gb,
            gpu_count=self.gpu_count,
            feasible=feasible,
            bottleneck=bottleneck,
            cost_usd=cost_usd,
            utilization_gpu=utilization_gpu,
            per_step=steps_data,
            metadata={
                "prefill_tps": 1.0 / self.prefill_time_per_token if self.prefill_time_per_token > 0 else 0.0,
                "decode_tps": 1.0 / self.decode_time_per_token if self.decode_time_per_token > 0 else 0.0,
                "kv_hbm_peak_gb": kv_hbm_peak_gb,
                "kv_hbf_peak_gb": kv_hbf_peak_gb,
                "total_llm_time_s": total_llm_time,
                "total_tool_time_s": total_tool_time,
                "env_init_time_s": env_init_time,
            },
        )
        return result

    def _identify_bottleneck(
        self,
        llm_time: float,
        tool_time: float,
        total_time: float,
        feasible: bool,
        memory_required_gb: float,
        available_memory_gb: float,
    ) -> str:
        """Classify the primary bottleneck."""
        if not feasible:
            if memory_required_gb > available_memory_gb:
                return "memory: HBM overflow"
            return "infeasible: unknown"

        if total_time <= 0:
            return "none"

        llm_ratio = llm_time / total_time
        tool_ratio = tool_time / total_time

        if tool_ratio > 0.5 and tool_ratio > llm_ratio:
            return "io-amplification: tool time dominates"
        if llm_ratio > 0.7:
            if llm_time > 0 and tool_time / llm_time > 6:
                return "io-amplification: tool time dominates"
            return "gpu: LLM time dominates"
        if tool_ratio > 0.3:
            return "cpu: tool/compile time significant"
        return "balanced"


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Convenience function to run a single simulation."""
    return SimulationEngine(config).run()
