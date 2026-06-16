# Multi-Layer Profiling

## Overview

The profiling framework decomposes a simulation result or real agent trace into four layers:

| Layer | Focus | Input |
|-------|-------|-------|
| **Software** | Tool calls, token distributions, step structure | `SimulationResult.per_step` or `trace.jsonl` |
| **Hardware** | FLOPs, memory bandwidth, utilization, cost | `SimulationResult` + `HardwareSpec` |
| **Algorithm** | FLOPs/token, KV-cache behavior, attention complexity | `ModelSpec` + `AlgorithmFamily` |
| **System** | Sandbox overhead, harness concurrency, parallelism | `SimulationResult` + `AgentHarnessSpec` |

The orchestrator correlates layers and produces actionable recommendations.

## CLI

```bash
# Profile a real trace
agent-sim profile \
  --trace trace.jsonl \
  --hardware H100-SXM5 \
  --model 70B-Dense \
  --output profile.md

# Profile a saved simulation result
agent-sim profile --input result.json --output profile.md
```

## Programmatic Use

```python
from agent_sim_platform.profiling import ProfilingOrchestrator
from agent_sim_platform.simulation import run_simulation

result = run_simulation(config)
orchestrator = ProfilingOrchestrator()
report = orchestrator.profile_simulation(result)

print(report.layers["hardware"]["utilization_gpu"])
print(report.recommendations)
```

## Cross-Layer Correlations

The orchestrator computes:

- `tool_time_vs_gpu_util`: high tool fraction may mask low GPU utilization.
- `memory_pressure`: peak memory vs. HBM capacity.
- `kv_vs_memory_bw`: KV bytes per token vs. available memory bandwidth.
- `comm_vs_compute`: communication time vs. compute time.

## Recommendations

Typical recommendations include:

- Reduce sandbox/tool latency when tool time dominates.
- Enable ZeRO-3 / TP / PP / quantization when memory pressure > 90%.
- Increase batch size when memory utilization is low.
- Adjust DP/TP/PP ratio when communication time exceeds 50% of compute.
- Limit batch size or enable chunked prefill when TTFT is high.

## Trace Format

The trace ingestor accepts `trace.jsonl` from `swe_bench_profiler`. Each line is a JSON object with fields such as:

```json
{
  "hardware": "H100-SXM5",
  "model": "70B-Dense",
  "steps": [
    {
      "llm_call": {
        "input_tokens": 1000,
        "output_tokens": 200,
        "prefill_time_ms": 50.0,
        "decode_time_ms": 800.0
      },
      "tool_call": {
        "name": "view",
        "execution_time_ms": 30.0
      },
      "context_state": {
        "kv_cache_tokens": 1200
      }
    }
  ]
}
```

Missing fields are treated as zero with a warning.
