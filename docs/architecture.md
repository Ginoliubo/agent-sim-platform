# Architecture

## Overview

Agent Sim Platform unifies three previously separate assets:

- `swe_bench_profiler/` — real-world profiling probes
- `ai-expert/simulation/` — Monte Carlo and feasibility simulators
- `ai-knowledge-base/` — hardware reference documents

The platform exposes them through a single Python package and CLI.

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Interface                                              │
│  CLI (agent-sim run/sweep/capacity/compare/analyze/report)   │
├─────────────────────────────────────────────────────────────┤
│  Simulation & Analysis                                       │
│  Engine / Sweep / Comparator / CapacityEstimator / Analyzer  │
├─────────────────────────────────────────────────────────────┤
│  Domain Libraries                                            │
│  Hardware Registry │ Model Registry │ Workload Registry      │
│  GPU │ NPU │ TPU  │ Dense │ MoE    │ SWE-agent │ Multi-agent │
├─────────────────────────────────────────────────────────────┤
│  Core Data Models & Utils                                    │
│  HardwareSpec / ModelSpec / WorkloadSpec / SimulationResult  │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Registries for extensibility**: new hardware, models, and workloads are added by registering presets.
2. **Unified `SimulationConfig`**: every run combines hardware, model, workload, harness, and optimization in one immutable input.
3. **Active vs total parameters**: MoE models use active parameters for compute and total parameters for weight memory.
4. **Future specs are explicit**: all unreleased accelerators are marked `is_future=True` with an uncertainty range.
5. **Backward compatibility**: existing `swe_bench_profiler/` scripts are preserved; the platform reads their trace format.

## Simulation Flow

1. **Derived quantities**: GPU count, prefill/decode throughput, KV bytes/token.
2. **Step loop**: sample tools and tokens, compute prefill/decode/tool times, update KV cache.
3. **HBM/HBF split**: if HBF offload is enabled, KV cache is split between HBM and HBF.
4. **Feasibility**: total required memory must fit within allocated GPUs at a max utilization threshold.
5. **Bottleneck**: classify as memory, GPU, CPU/tool, IO amplification, or balanced.
6. **Cost**: GPU rental cost based on runtime and hourly rate.

## Extending the Platform

- Add hardware: see [adding-hardware.md](adding-hardware.md).
- Add workload: subclass `WorkloadSpec` and register it.
- Add optimization: create a new `OptimizationConfig` preset.
- Add analysis: implement an analyzer in `analysis/`.
