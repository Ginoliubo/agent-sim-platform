# Training Simulation

## Overview

The training engine simulates distributed training jobs, modeling compute, communication, and memory.

## Supported Strategies

- `pretrain`
- `sft`
- `rlhf`
- `dpo`
- `grpo`

## Memory Model

Per-GPU memory includes:

- Weights (sharded by TP/PP/EP/ZeRO-3)
- Gradients (sharded by ZeRO-2/3)
- Optimizer states (AdamW = 8 bytes/param; sharded by ZeRO-1/2/3)
- Activations (reduced by gradient checkpointing)
- KV cache for the full training sequence

## Communication Model

- Data Parallel: Ring All-Reduce for gradients
- Tensor Parallel: All-Reduce per layer
- Pipeline Parallel: bubble overhead
- MoE Expert Parallel: All-to-All for token routing

## CLI

```bash
agent-sim train \
  --hardware H100-SXM5 \
  --model 70B-Dense \
  --algorithm dense \
  --dataset-tokens 1T \
  --dp 64 --tp 8 --pp 4 \
  --mfu-target 0.35
```

## Output Metrics

- `total_time_seconds`: end-to-end training time
- `step_time_seconds`: average time per global step
- `compute_time_seconds`: pure GPU compute time
- `communication_time_seconds`: communication overhead
- `memory_per_gpu_gb`: memory required per GPU
- `mfu`: model FLOPs utilization
- `cost_usd`: estimated rental cost

## Programmatic Use

```python
from agent_sim_platform.data_models import TrainingConfig, ParallelismConfig
from agent_sim_platform.hardware import DEFAULT_REGISTRY as HW_REGISTRY
from agent_sim_platform.models import DEFAULT_REGISTRY as MODEL_REGISTRY
from agent_sim_platform.simulation.training import run_training

result = run_training(
    MODEL_REGISTRY.get("70B-Dense"),
    HW_REGISTRY.get("H100-SXM5"),
    TrainingConfig(
        strategy="pretrain",
        dataset_tokens=1_000_000_000_000,
        parallelism=ParallelismConfig(dp=64, tp=8, pp=4),
    ),
)
```
