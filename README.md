# Agent Sim Platform

Full-stack simulation platform for AI agent workloads, training, and inference serving on AI infrastructure (GPU / NPU / TPU).

## Quick Start

```bash
# From the repo root
cd agent-sim-platform

# Run a single agent simulation
python3 -m agent_sim_platform run \
  --hardware H100-SXM5 \
  --model 1T-MoE \
  --workload swe-agent \
  --context 32K \
  --optimization layer2

# Training simulation
python3 -m agent_sim_platform train \
  --hardware H100-SXM5 \
  --model 70B-Dense \
  --algorithm dense \
  --dataset-tokens 1T \
  --dp 64 --tp 8 --pp 4

# Inference serving simulation
python3 -m agent_sim_platform serve \
  --hardware H100-SXM5 \
  --model 70B-Dense \
  --algorithm dense \
  --arrival-rate 10 \
  --max-batch-size 32 \
  --gpu-count 8

# Compare hardware
python3 -m agent_sim_platform compare-hardware \
  --hardware H100-SXM5,B200,Rubin \
  --model 10T-MoE \
  --context 1M

# Capacity estimation
python3 -m agent_sim_platform capacity \
  --hardware Rubin-Ultra \
  --model 10T-MoE \
  --context 10M \
  --precision FP4
```

## Installation

```bash
python3 -m pip install -e .
```

This installs the `agent-sim` CLI entry point.

## Supported Hardware

| Vendor  | Current                     | Future (estimated)              |
|---------|----------------------------|---------------------------------|
| NVIDIA  | A100, H100, B200           | Rubin, Rubin Ultra, Feynman     |
| Huawei  | Ascend 910B                | Ascend 950, 960, 970            |
| Google  | TPU v5e, v5p, v6e          | TPU v7                          |

Future specs are flagged with `is_future=True` and an uncertainty range.

## Supported Models

- 70B-Dense, 300B-Dense, 1T-Dense
- 1T-MoE, 10T-MoE, 30T-MoE

## Supported Algorithm Families

- `dense`: standard dense transformer
- `moe`: Mixture-of-Experts
- `mamba`: State Space Model (no KV cache, O(n))
- `linear_attention`: kernelized linear attention
- `ring_attention`: sequence-parallel chunked attention

## Supported Workloads

### Agent Workloads
- `swe-agent`: SWE-bench style ReAct loop
- `claude-code`: IDE-integrated coding assistant
- `multi-agent`: parallel agent swarm

### Training Workloads
- `pretrain`, `sft`, `rlhf`, `grpo`

### Inference Serving Workloads
- `chat`, `code-completion`, `long-context`

## CLI Reference

```bash
agent-sim run                 # Single agent simulation
agent-sim sweep               # Grid sweep
agent-sim capacity            # Capacity estimation
agent-sim train               # Training simulation
agent-sim serve               # Inference serving simulation
agent-sim compare-hardware
agent-sim analyze-trace
agent-sim report
agent-sim list-hardware
agent-sim list-models
agent-sim list-workloads
agent-sim list-algorithms
```

## Project Structure

```
agent_sim_platform/
├── algorithms/    # Algorithm family presets (Dense/MoE/Mamba/...)
├── hardware/      # GPU / NPU / TPU presets and registry
├── models/        # Model presets and registry
├── workloads/     # Agent, training, and inference workload presets
├── simulation/    # Engine, runner, sweep, capacity, training, serving
├── analysis/      # Bottleneck, cost, and training cost analysis
├── reports/       # JSON / Markdown reporters
├── profiling/     # Bridge to swe_bench_profiler traces
└── cli.py         # Unified CLI
```

## Documentation

- [Architecture](docs/architecture.md)
- [Data Models](docs/data-models.md)
- [Adding Hardware](docs/adding-hardware.md)
- [Algorithms](docs/algorithms.md)
- [Training Simulation](docs/training.md)
- [Inference Serving](docs/inference-serving.md)

## Running Tests

```bash
python3 -m pytest tests/ -q
```

## License

MIT
