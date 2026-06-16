"""Bottleneck diagnosis aligned with analyze_trace.py categories."""

from ..config import BOTTLENECK_THRESHOLDS
from ..data_models import SimulationResult


class BottleneckAnalyzer:
    """Diagnose bottlenecks from a SimulationResult."""

    def __init__(self, result: SimulationResult):
        self.result = result

    def diagnose(self) -> str:
        """Return the primary bottleneck label."""
        return self.result.bottleneck

    def details(self) -> dict:
        """Return detailed bottleneck flags."""
        r = self.result
        cfg = r.config
        thresholds = BOTTLENECK_THRESHOLDS

        total_time = r.latency_seconds
        llm_time = r.metadata.get("total_llm_time_s", 0.0)
        tool_time = r.metadata.get("total_tool_time_s", 0.0)

        gpu_util = r.utilization_gpu
        memory_ratio = (
            r.memory_required_gb / (cfg.hardware.memory_gb * r.gpu_count)
            if r.gpu_count > 0 else 1.0
        )

        io_amp = tool_time / llm_time if llm_time > 0 else float("inf")

        return {
            "memory_overflow": not r.feasible and memory_ratio > 1.0,
            "gpu_utilization_low": gpu_util < thresholds["gpu_utilization"]["warning"],
            "memory_pressure": memory_ratio > thresholds["gpu_memory"]["warning"],
            "io_amplification": io_amp > thresholds["io_amplification"]["warning"],
            "io_amplification_critical": io_amp > thresholds["io_amplification"]["critical"],
            "cost_high": r.cost_usd > thresholds["cost_per_task"]["warning"],
            "cost_critical": r.cost_usd > thresholds["cost_per_task"]["critical"],
            "primary": r.bottleneck,
        }


def diagnose(result: SimulationResult) -> str:
    """Convenience function returning the primary bottleneck."""
    return BottleneckAnalyzer(result).diagnose()
