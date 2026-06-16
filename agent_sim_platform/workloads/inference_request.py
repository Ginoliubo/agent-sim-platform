"""Inference service workload specification."""

from dataclasses import dataclass, field
from typing import Dict

from ..data_models import InferenceServiceConfig


@dataclass
class InferenceRequestWorkload:
    """A workload definition for inference serving."""

    name: str
    service_config: InferenceServiceConfig
    description: str = ""


CHAT_WORKLOAD = InferenceRequestWorkload(
    name="chat",
    service_config=InferenceServiceConfig(
        arrival_rate_per_sec=10.0,
        arrival_distribution="poisson",
        target_ttft_ms=2000.0,
        target_tpot_ms=50.0,
        max_batch_size=64,
        max_queue_len=32,
        request_length_mean=4096,
        request_length_std=2048,
        output_length_mean=512,
        output_length_std=256,
        simulation_duration_seconds=60.0,
    ),
    description="Chat-like inference workload with moderate context and output.",
)

CODE_COMPLETION_WORKLOAD = InferenceRequestWorkload(
    name="code-completion",
    service_config=InferenceServiceConfig(
        arrival_rate_per_sec=50.0,
        arrival_distribution="poisson",
        target_ttft_ms=500.0,
        target_tpot_ms=20.0,
        max_batch_size=128,
        max_queue_len=64,
        request_length_mean=2048,
        request_length_std=1024,
        output_length_mean=128,
        output_length_std=64,
        simulation_duration_seconds=60.0,
    ),
    description="High-rate code completion with short inputs and outputs.",
)

LONG_CONTEXT_WORKLOAD = InferenceRequestWorkload(
    name="long-context",
    service_config=InferenceServiceConfig(
        arrival_rate_per_sec=2.0,
        arrival_distribution="poisson",
        target_ttft_ms=10000.0,
        target_tpot_ms=100.0,
        max_batch_size=16,
        max_queue_len=16,
        request_length_mean=32768,
        request_length_std=16384,
        output_length_mean=1024,
        output_length_std=512,
        simulation_duration_seconds=60.0,
    ),
    description="Long-context summarization or analysis workload.",
)

INFERENCE_WORKLOAD_PRESETS = [CHAT_WORKLOAD, CODE_COMPLETION_WORKLOAD, LONG_CONTEXT_WORKLOAD]


class InferenceWorkloadRegistry:
    """Registry for inference workload presets."""

    def __init__(self, workloads=None):
        self._workloads = {}
        if workloads:
            for w in workloads:
                self.register(w)

    def register(self, workload: InferenceRequestWorkload) -> None:
        if workload.name in self._workloads:
            raise ValueError(f"Inference workload '{workload.name}' already registered")
        self._workloads[workload.name] = workload

    def get(self, name: str) -> InferenceRequestWorkload:
        if name not in self._workloads:
            raise KeyError(f"Inference workload '{name}' not found")
        return self._workloads[name]

    def list(self):
        return sorted(self._workloads.values(), key=lambda w: w.name)

    def names(self):
        return sorted(self._workloads.keys())


DEFAULT_REGISTRY = InferenceWorkloadRegistry(INFERENCE_WORKLOAD_PRESETS)

__all__ = [
    "InferenceRequestWorkload",
    "CHAT_WORKLOAD",
    "CODE_COMPLETION_WORKLOAD",
    "LONG_CONTEXT_WORKLOAD",
    "INFERENCE_WORKLOAD_PRESETS",
    "InferenceWorkloadRegistry",
    "DEFAULT_REGISTRY",
]
