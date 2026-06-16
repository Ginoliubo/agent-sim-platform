"""Bridge between swe_bench_profiler outputs and SimulationResult."""

import json
from pathlib import Path
from typing import Dict, List

from ..data_models import (
    AgentHarnessSpec,
    HardwareSpec,
    ModelSpec,
    OptimizationConfig,
    SimulationConfig,
    SimulationResult,
    WorkloadSpec,
)
from ..hardware import DEFAULT_REGISTRY as HW_REGISTRY


class TraceAnalyzer:
    """Read and summarize a legacy trace.jsonl file."""

    def __init__(self, trace_path: str):
        self.trace_path = Path(trace_path)
        self.records: List[Dict] = []
        self._load()

    def _load(self) -> None:
        with self.trace_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def analyze(self) -> SimulationResult:
        """Aggregate trace records into a SimulationResult."""
        if not self.records:
            raise ValueError("No trace records found")

        # Use first record for config inference
        first = self.records[0]
        steps = first.get("steps", [])

        total_llm_time = 0.0
        total_tool_time = 0.0
        total_prefill_tokens = 0
        total_decode_tokens = 0
        peak_kv_gb = 0.0
        per_step = []

        for idx, step in enumerate(steps, 1):
            llm_call = step.get("llm_call", {})
            tool_call = step.get("tool_call", {})
            context_state = step.get("context_state", {})

            prefill_tok = llm_call.get("input_tokens", 0)
            decode_tok = llm_call.get("output_tokens", 0)
            prefill_ms = llm_call.get("prefill_time_ms", 0.0)
            decode_ms = llm_call.get("decode_time_ms", 0.0)
            tool_ms = tool_call.get("execution_time_ms", 0.0)
            kv_gb = context_state.get("kv_cache_tokens", 0) * 0.001  # placeholder

            total_prefill_tokens += prefill_tok
            total_decode_tokens += decode_tok
            total_llm_time += (prefill_ms + decode_ms) / 1000.0
            total_tool_time += tool_ms / 1000.0
            peak_kv_gb = max(peak_kv_gb, kv_gb)

            per_step.append(
                {
                    "step": idx,
                    "tool": tool_call.get("name", "unknown"),
                    "prefill_input": prefill_tok,
                    "decode_output": decode_tok,
                    "prefill_time_ms": prefill_ms,
                    "decode_time_ms": decode_ms,
                    "tool_time_ms": tool_ms,
                    "kv_total_gb": kv_gb,
                }
            )

        latency = total_llm_time + total_tool_time

        # Infer hardware from trace metadata if possible
        hw_name = first.get("hardware", "H100-SXM5")
        try:
            hardware = HW_REGISTRY.get(hw_name)
        except KeyError:
            hardware = HW_REGISTRY.get("H100-SXM5")

        config = SimulationConfig(
            hardware=hardware,
            model=ModelSpec(
                name=first.get("model", "unknown"),
                total_params_b=first.get("params_b", 100),
                active_params_b=first.get("active_params_b", 100),
                n_layers=first.get("n_layers", 80),
                d_model=first.get("d_model", 8192),
                n_heads=first.get("n_heads", 64),
                d_head=first.get("d_head", 128),
            ),
            workload=WorkloadSpec(name="trace", max_steps=1, avg_steps=1.0, step_std=0.0, context_limit=128000),
            harness=AgentHarnessSpec(name="trace"),
        )

        return SimulationResult(
            config=config,
            latency_seconds=latency,
            tokens_total=total_prefill_tokens + total_decode_tokens,
            tokens_input=total_prefill_tokens,
            tokens_output=total_decode_tokens,
            peak_kv_gb=peak_kv_gb,
            feasible=True,
            bottleneck="trace-replay",
            per_step=per_step,
            metadata={
                "trace_records": len(self.records),
                "total_llm_time_s": total_llm_time,
                "total_tool_time_s": total_tool_time,
            },
        )
