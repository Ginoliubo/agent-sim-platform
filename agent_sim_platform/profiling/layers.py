"""Multi-layer profilers for simulation results and traces."""

from typing import Dict, List, Optional

from ..data_models import ModelSpec, SimulationResult


class SoftwareLayerProfiler:
    """Profile software-level behavior from per-step data or trace records."""

    def profile_simulation(self, result: SimulationResult) -> Dict:
        """Profile from a SimulationResult's per_step metadata."""
        steps = result.per_step or []
        if not steps:
            return {"samples": 0, "notes": "No per-step data available"}

        tool_counts: Dict[str, int] = {}
        total_tool_time_ms = 0.0
        total_llm_time_ms = 0.0
        total_prefill_tokens = 0
        total_decode_tokens = 0

        for step in steps:
            tool_name = step.get("tool", "unknown")
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            total_tool_time_ms += step.get("tool_time_ms", 0.0)
            total_llm_time_ms += step.get("prefill_time_ms", 0.0) + step.get(
                "decode_time_ms", 0.0
            )
            total_prefill_tokens += step.get("prefill_input", 0)
            total_decode_tokens += step.get("decode_output", 0)

        total_time_ms = total_tool_time_ms + total_llm_time_ms
        return {
            "samples": len(steps),
            "tool_counts": tool_counts,
            "total_prefill_tokens": total_prefill_tokens,
            "total_decode_tokens": total_decode_tokens,
            "total_llm_time_ms": total_llm_time_ms,
            "total_tool_time_ms": total_tool_time_ms,
            "tool_time_fraction": total_tool_time_ms / total_time_ms if total_time_ms > 0 else 0.0,
            "llm_time_fraction": total_llm_time_ms / total_time_ms if total_time_ms > 0 else 0.0,
        }

    def profile_trace_records(self, records: List[Dict]) -> Dict:
        """Profile from raw swe_bench_profiler trace records."""
        if not records:
            return {"samples": 0, "notes": "No trace records"}

        # Use first record for step aggregation
        first = records[0]
        steps = first.get("steps", [])
        adapted = []
        for step in steps:
            llm_call = step.get("llm_call", {})
            tool_call = step.get("tool_call", {})
            adapted.append(
                {
                    "tool": tool_call.get("name", "unknown"),
                    "prefill_input": llm_call.get("input_tokens", 0),
                    "decode_output": llm_call.get("output_tokens", 0),
                    "prefill_time_ms": llm_call.get("prefill_time_ms", 0.0),
                    "decode_time_ms": llm_call.get("decode_time_ms", 0.0),
                    "tool_time_ms": tool_call.get("execution_time_ms", 0.0),
                }
            )

        dummy = SimulationResult(
            config=first.get("config"),
            per_step=adapted,
        )
        return self.profile_simulation(dummy)


class HardwareLayerProfiler:
    """Profile hardware utilization and efficiency."""

    def profile(self, result: SimulationResult) -> Dict:
        cfg = result.config
        hw = cfg.hardware
        total_flops = result.metadata.get("total_flops", 0.0)
        peak_compute_flops = hw.effective_flops(cfg.precision, 1.0) * result.gpu_count
        compute_efficiency = (
            total_flops / (peak_compute_flops * result.latency_seconds)
            if peak_compute_flops > 0 and result.latency_seconds > 0
            else 0.0
        )

        memory_peak_gb = max(result.peak_kv_gb, result.memory_required_gb)
        memory_utilization = (
            memory_peak_gb / (hw.memory_gb * result.gpu_count)
            if hw.memory_gb > 0 and result.gpu_count > 0
            else 0.0
        )

        return {
            "gpu_count": result.gpu_count,
            "hardware": hw.name,
            "precision": cfg.precision,
            "total_flops": total_flops,
            "peak_compute_flops": peak_compute_flops,
            "compute_efficiency": compute_efficiency,
            "utilization_gpu": result.utilization_gpu,
            "memory_peak_gb": memory_peak_gb,
            "memory_capacity_gb": hw.memory_gb * result.gpu_count,
            "memory_utilization": memory_utilization,
            "memory_bw_tb_s": hw.memory_bw_tb_s,
            "interconnect_bw_gb_s": hw.interconnect_bw_gb_s,
            "cost_usd": result.cost_usd,
            "cost_per_million_tokens": (
                result.cost_usd / (result.tokens_total / 1e6)
                if result.tokens_total > 0
                else 0.0
            ),
        }


class AlgorithmLayerProfiler:
    """Profile algorithm-level characteristics."""

    def profile(self, model: ModelSpec, precision: str = "FP8") -> Dict:
        family = model.algorithm_family
        return {
            "model_name": model.name,
            "algorithm_family": family.name if family else "unknown",
            "attention_complexity": family.attention_complexity if family else "unknown",
            "has_kv_cache": family.has_kv_cache if family else False,
            "kv_scaling": family.kv_scaling if family else "unknown",
            "total_params_b": model.total_params_b,
            "active_params_b": model.active_params_b,
            "active_ratio": (
                model.active_params_b / model.total_params_b
                if model.total_params_b > 0
                else 0.0
            ),
            "flops_per_token_forward": model.flops_per_token_forward(),
            "flops_per_token_backward": model.flops_per_token_backward(),
            "flops_per_token_training": model.flops_per_token_training(),
            "kv_bytes_per_token": model.kv_bytes_per_token(precision),
            "n_layers": model.n_layers,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "d_head": model.d_head,
            "num_experts": model.num_experts,
            "top_k": model.top_k,
        }


class SystemLayerProfiler:
    """Profile system-level overhead and parallelism."""

    def profile(self, result: SimulationResult) -> Dict:
        cfg = result.config
        harness = cfg.harness
        metadata = result.metadata

        compute_time = metadata.get("compute_time_seconds", 0.0)
        communication_time = metadata.get("communication_time_seconds", 0.0)
        step_time = metadata.get("step_time_seconds", 0.0)

        parallelism = getattr(metadata, "parallelism", {}) or {}
        # If training result, try to extract parallelism from config (stored in metadata)
        if "dp" not in parallelism and hasattr(cfg, "tp"):
            parallelism = {
                "tp": cfg.tp,
                "pp": cfg.pp,
                "batch_size": cfg.batch_size,
            }

        return {
            "harness": cfg.harness.name,
            "concurrency": harness.concurrency,
            "sandbox_type": harness.sandbox_type,
            "control_cpu_percent": harness.control_cpu_percent,
            "parallelism": parallelism,
            "compute_time_seconds": compute_time,
            "communication_time_seconds": communication_time,
            "step_time_seconds": step_time,
            "communication_ratio": (
                communication_time / compute_time if compute_time > 0 else 0.0
            ),
            "bottleneck": result.bottleneck,
            "feasible": result.feasible,
        }


__all__ = [
    "AlgorithmLayerProfiler",
    "HardwareLayerProfiler",
    "SoftwareLayerProfiler",
    "SystemLayerProfiler",
]
