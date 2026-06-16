# Inference Serving Simulation

## Overview

The inference serving engine simulates a request-driven inference service with continuous batching.

## Model

- Requests arrive according to a Poisson or fixed distribution
- Each request has an input length (prefill) and output length (decode)
- New requests are batched for prefill
- Ongoing requests continue decoding
- Memory limits and max batch size constrain admission

## Output Metrics

- `requests_total`: total requests in simulation window
- `requests_completed`: successfully completed
- `requests_dropped`: dropped due to queue overflow or memory
- `throughput_req_per_sec`: request throughput
- `throughput_tok_per_sec`: token throughput
- `ttft_p50_ms` / `ttft_p99_ms`: time to first token
- `tpot_p50_ms` / `tpot_p99_ms`: time per output token
- `e2e_latency_p99_ms`: end-to-end latency p99

## CLI

```bash
agent-sim serve \
  --hardware H100-SXM5 \
  --model 70B-Dense \
  --algorithm dense \
  --arrival-rate 10 \
  --max-batch-size 32 \
  --gpu-count 8
```

## Programmatic Use

```python
from agent_sim_platform.data_models import InferenceServiceConfig
from agent_sim_platform.hardware import DEFAULT_REGISTRY as HW_REGISTRY
from agent_sim_platform.models import DEFAULT_REGISTRY as MODEL_REGISTRY
from agent_sim_platform.simulation.inference_serving import run_serving

result = run_serving(
    MODEL_REGISTRY.get("70B-Dense"),
    HW_REGISTRY.get("H100-SXM5"),
    InferenceServiceConfig(arrival_rate_per_sec=10.0, max_batch_size=32),
    gpu_count=8,
)
```
