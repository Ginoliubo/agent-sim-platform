"""Profiling orchestrator: correlate software/hardware/algorithm/system layers."""

from pathlib import Path
from typing import Dict, List, Optional

from ..data_models import ProfilingReport, SimulationResult
from ..hardware import DEFAULT_REGISTRY as HW_REGISTRY
from ..models import DEFAULT_REGISTRY as MODEL_REGISTRY
from .ingestors import SimulationIngestor, TraceIngestor
from .layers import (
    AlgorithmLayerProfiler,
    HardwareLayerProfiler,
    SoftwareLayerProfiler,
    SystemLayerProfiler,
)


class ProfilingOrchestrator:
    """Run multi-layer profiling on a simulation result or trace."""

    def __init__(self):
        self.software_profiler = SoftwareLayerProfiler()
        self.hardware_profiler = HardwareLayerProfiler()
        self.algorithm_profiler = AlgorithmLayerProfiler()
        self.system_profiler = SystemLayerProfiler()

    def profile_simulation(self, result: SimulationResult) -> ProfilingReport:
        """Profile a SimulationResult across all layers."""
        report = ProfilingReport()
        report.layers["software"] = self.software_profiler.profile_simulation(result)
        report.layers["hardware"] = self.hardware_profiler.profile(result)
        report.layers["algorithm"] = self.algorithm_profiler.profile(
            result.config.model, result.config.precision
        )
        report.layers["system"] = self.system_profiler.profile(result)
        report.correlations = self._compute_correlations(report)
        report.recommendations = self._generate_recommendations(report)
        report.raw_samples = result.per_step or []
        return report

    def profile_trace(
        self,
        trace_path: str,
        hardware_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> ProfilingReport:
        """Profile a legacy trace.jsonl across all layers."""
        ingestor = TraceIngestor(trace_path)
        result = ingestor.ingest()

        # Override inferred hardware/model if requested
        if hardware_name:
            result.config.hardware = HW_REGISTRY.get(hardware_name)
        if model_name:
            result.config.model = MODEL_REGISTRY.get(model_name)

        return self.profile_simulation(result)

    def compare(
        self,
        simulation_result: SimulationResult,
        trace_path: str,
    ) -> Dict:
        """Compare a simulation result against a real trace profile."""
        sim_report = self.profile_simulation(simulation_result)
        trace_report = self.profile_trace(trace_path)

        return {
            "software": {
                "sim_tool_time_fraction": sim_report.layers["software"].get(
                    "tool_time_fraction", 0.0
                ),
                "trace_tool_time_fraction": trace_report.layers["software"].get(
                    "tool_time_fraction", 0.0
                ),
            },
            "hardware": {
                "sim_utilization": sim_report.layers["hardware"].get(
                    "utilization_gpu", 0.0
                ),
                "trace_utilization": trace_report.layers["hardware"].get(
                    "utilization_gpu", 0.0
                ),
            },
            "recommendations": list(
                set(sim_report.recommendations + trace_report.recommendations)
            ),
        }

    def _compute_correlations(self, report: ProfilingReport) -> Dict:
        """Compute cross-layer correlations."""
        sw = report.layers.get("software", {})
        hw = report.layers.get("hardware", {})
        algo = report.layers.get("algorithm", {})
        sys = report.layers.get("system", {})

        return {
            "tool_time_vs_gpu_util": {
                "tool_time_fraction": sw.get("tool_time_fraction", 0.0),
                "gpu_utilization": hw.get("utilization_gpu", 0.0),
            },
            "memory_pressure": {
                "memory_utilization": hw.get("memory_utilization", 0.0),
                "memory_peak_gb": hw.get("memory_peak_gb", 0.0),
                "memory_capacity_gb": hw.get("memory_capacity_gb", 0.0),
            },
            "kv_vs_memory_bw": {
                "kv_bytes_per_token": algo.get("kv_bytes_per_token", 0.0),
                "memory_bw_tb_s": hw.get("memory_bw_tb_s", 0.0),
            },
            "comm_vs_compute": {
                "communication_ratio": sys.get("communication_ratio", 0.0),
                "compute_time_seconds": sys.get("compute_time_seconds", 0.0),
            },
        }

    def _generate_recommendations(self, report: ProfilingReport) -> List[str]:
        """Generate optimization recommendations from layer data."""
        recs = []
        sw = report.layers.get("software", {})
        hw = report.layers.get("hardware", {})
        algo = report.layers.get("algorithm", {})
        sys = report.layers.get("system", {})

        if sw.get("tool_time_fraction", 0.0) > 0.5:
            recs.append(
                "Tool execution dominates wall time: optimize sandbox/tool latency or reduce tool calls."
            )

        if sw.get("total_decode_tokens", 0) > sw.get("total_prefill_tokens", 0) * 2:
            recs.append(
                "Decode tokens significantly exceed prefill tokens: consider KV-cache optimizations."
            )

        memory_util = hw.get("memory_utilization", 0.0)
        if memory_util > 0.9:
            recs.append(
                "Memory utilization >90%: enable ZeRO-3 / TP / PP / quantization or reduce batch size."
            )
        elif memory_util < 0.3:
            recs.append(
                "Memory utilization <30%: increase batch size or use larger context to improve throughput."
            )

        if algo.get("active_ratio", 1.0) < 0.5 and algo.get("has_kv_cache", True):
            recs.append(
                "MoE architecture with high total/active param ratio: ensure EP size matches expert count."
            )

        comm_ratio = sys.get("communication_ratio", 0.0)
        if comm_ratio > 0.5:
            recs.append(
                "Communication time >50% of compute: reduce DP degree, increase TP/PP, or upgrade interconnect."
            )

        if sys.get("bottleneck", "").startswith("memory"):
            recs.append("Bottleneck is memory capacity: shard weights/optimizer states or quantize.")

        if not recs:
            recs.append("No major cross-layer anomaly detected.")

        return recs


__all__ = ["ProfilingOrchestrator"]
