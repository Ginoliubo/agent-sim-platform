# Agent Sim Platform

Full-stack simulation platform for AI agent workloads and AI infrastructure (GPU / NPU / TPU).

## Quick Start

```bash
# From the repo root
cd agent-sim-platform

# Run a single simulation
python3 -m agent_sim_platform run \
  --hardware H100-SXM5 \
  --model 1T-MoE \
  --workload swe-agent \
  --context 32K \
  --optimization layer2

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

## Supported Workloads

- `swe-agent`: SWE-bench style ReAct loop
- `claude-code`: IDE-integrated coding assistant
- `multi-agent`: parallel agent swarm

## CLI Reference

```bash
agent-sim run          # Single simulation
agent-sim sweep        # Grid sweep
agent-sim capacity     # Capacity estimation
agent-sim compare-hardware
agent-sim analyze-trace
agent-sim report
agent-sim list-hardware
agent-sim list-models
agent-sim list-workloads
```

## Project Structure

```
agent_sim_platform/
├── hardware/      # GPU / NPU / TPU presets and registry
├── models/        # Model presets and registry
├── workloads/     # Workload presets and registry
├── simulation/    # Engine, runner, sweep, capacity estimator
├── analysis/      # Bottleneck and cost analysis
├── reports/       # JSON / Markdown reporters
├── profiling/     # Bridge to swe_bench_profiler traces
└── cli.py         # Unified CLI
```

## Documentation

- [Architecture](docs/architecture.md)
- [Data Models](docs/data-models.md)
- [Adding Hardware](docs/adding-hardware.md)

## Running Tests

```bash
python3 -m pytest tests/ -q
```

## License

MIT
