# Benchmark Fixtures

## Overview

`agent-sim-platform` ships with a set of **authoritative benchmark fixtures** drawn from public papers and reproducible community benchmarks. These fixtures capture known input configurations and observed output metrics, enabling the simulator to be calibrated against reality.

## Built-in Fixtures

| Fixture | Domain | Source | Key Metrics |
|---------|--------|--------|-------------|
| `llama2_70b_pretrain` | training | LLaMA-2 paper | 2T tokens, ~1.7M A100 GPU-hours |
| `llama3_70b_pretrain` | training | LLaMA-3 paper | 15T tokens, ~4M H100 GPU-hours |
| `deepseek_v3_pretrain` | training | DeepSeek-V3 tech report | 14.8T tokens, 2.788M H800 GPU-hours |
| `vllm_llama70b_serving` | serving | vLLM paper / repro | TTFT / TPOT / throughput on ShareGPT |
| `mixtral_8x22b_serving` | serving | vLLM / SGLang community | MoE serving throughput on H100 |
| `capacity_llama70b_4k` | capacity | Public deployment notes | 70B FP16 fits on 2×A100-80GB |

Fixtures are stored as YAML files in `agent_sim_platform/benchmarks/fixtures/`.

## CLI

```bash
# List all fixtures
agent-sim list-benchmarks

# Filter by domain
agent-sim list-benchmarks --domain training

# Evaluate a single fixture
agent-sim benchmark --name llama2_70b_pretrain
```

## Fixture Format

```yaml
name: llama2_70b_pretrain
domain: training
source: "LLaMA 2: Open Foundation and Fine-Tuned Chat Models"
source_url: "https://arxiv.org/abs/2307.09288"
hardware_names:
  - A100-SXM4
model_name: 70B-Dense
algorithm_name: dense
config:
  strategy: pretrain
  dataset_tokens: 2000000000000
  global_batch_size: 976
  sequence_length: 4096
  zero_stage: 1
  mfu_target: 0.43
  parallelism:
    dp: 2048
    tp: 1
    pp: 1
observed_metrics:
  total_time_seconds: 3024000
  gpu_hours: 1720320
  mfu: 0.43
tolerance:
  mape: 0.25
notes: "Reported 1.7M A100-80GB GPU-hours for 2T tokens."
```

## Adding a Fixture

1. Create a new YAML file in `agent_sim_platform/benchmarks/fixtures/`.
2. Ensure `domain` is one of `training`, `serving`, or `capacity`.
3. Provide `hardware_names`, `model_name`, and `algorithm_name` that exist in the registries.
4. Document assumptions and source in `notes`.

The loader will automatically pick up the fixture on the next CLI invocation.
