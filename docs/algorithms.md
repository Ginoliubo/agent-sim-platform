# Algorithm Families

## Overview

Agent Sim Platform models five algorithm families, each with different attention complexity, KV cache behavior, and FLOP profile.

## Families

### `dense`

Standard dense Transformer with O(n²) attention.

- **KV cache**: yes, per-token
- **FLOPs/token forward**: 2 × P_active
- **FLOPs/token backward**: 4 × P_active
- **Typical optimizations**: FlashAttention, GQA, prefix caching

### `moe`

Mixture-of-Experts with sparse routing.

- **KV cache**: yes, per-token
- **FLOPs/token forward**: 2 × P_active
- **Weight memory**: total parameters
- **Typical optimizations**: Expert Parallelism (EP), top-k routing

### `mamba`

State Space Model (e.g., Mamba, RWKV-7). No attention matrix.

- **KV cache**: none
- **Attention complexity**: O(n)
- **FLOPs/token forward**: 2 × P
- **Best for**: very long context streaming

### `linear_attention`

Kernelized linear attention with compressed recurrent state.

- **KV cache**: yes, compressed (default 4x smaller)
- **Attention complexity**: O(n)
- **FLOPs/token forward**: 2 × P

### `ring_attention`

Sequence-parallel chunked attention for ultra-long context.

- **KV cache**: yes, distributed in chunks
- **Attention complexity**: O(n²) per chunk
- **FLOPs/token forward**: 2 × P

## CLI

```bash
agent-sim list-algorithms
```

## Programmatic Use

```python
from agent_sim_platform.algorithms import DEFAULT_REGISTRY
from agent_sim_platform.models import DEFAULT_REGISTRY as MODEL_REGISTRY

family = DEFAULT_REGISTRY.get("mamba")
model = MODEL_REGISTRY.get("70B-Dense")
kv = family.kv_bytes_per_token(model, "FP8")  # 0.0
flops = family.flops_per_token_forward(model)
```
