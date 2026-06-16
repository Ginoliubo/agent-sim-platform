# Profiling Report

## Software Layer

- **samples**: 2
- **tool_counts**: {'view': 1, 'bash_test': 1}
- **total_prefill_tokens**: 2200
- **total_decode_tokens**: 350
- **total_llm_time_ms**: 1510.0000
- **total_tool_time_ms**: 5030.0000
- **tool_time_fraction**: 0.7691
- **llm_time_fraction**: 0.2309

## Hardware Layer

- **gpu_count**: 1
- **hardware**: H100-SXM5
- **precision**: FP8
- **total_flops**: 0.0000
- **peak_compute_flops**: 1979000000000000.0000
- **compute_efficiency**: 0.0000
- **utilization_gpu**: 0.0000
- **memory_peak_gb**: 1.3500
- **memory_capacity_gb**: 80.0000
- **memory_utilization**: 0.0169
- **memory_bw_tb_s**: 3.3500
- **interconnect_bw_gb_s**: 900.0000
- **cost_usd**: 0.0000
- **cost_per_million_tokens**: 0.0000

## Algorithm Layer

- **model_name**: 70B-Dense
- **algorithm_family**: dense
- **attention_complexity**: quadratic
- **has_kv_cache**: True
- **kv_scaling**: per_token
- **total_params_b**: 70
- **active_params_b**: 70
- **active_ratio**: 1.0000
- **flops_per_token_forward**: 140000000000.0000
- **flops_per_token_backward**: 280000000000.0000
- **flops_per_token_training**: 420000000000.0000
- **kv_bytes_per_token**: 327680.0000
- **n_layers**: 80
- **d_model**: 8192
- **n_heads**: 64
- **d_head**: 128
- **num_experts**: 1
- **top_k**: 1

## System Layer

- **harness**: trace
- **concurrency**: 1
- **sandbox_type**: docker
- **control_cpu_percent**: 0.0500
- **parallelism**: {'tp': 8, 'pp': 1, 'batch_size': 1}
- **compute_time_seconds**: 0.0000
- **communication_time_seconds**: 0.0000
- **step_time_seconds**: 0.0000
- **communication_ratio**: 0.0000
- **bottleneck**: trace-replay
- **feasible**: True

## Cross-Layer Correlations

- **tool_time_vs_gpu_util**: {'tool_time_fraction': 0.7691131498470948, 'gpu_utilization': 0.0}
- **memory_pressure**: {'memory_utilization': 0.016875, 'memory_peak_gb': 1.35, 'memory_capacity_gb': 80.0}
- **kv_vs_memory_bw**: {'kv_bytes_per_token': 327680.0, 'memory_bw_tb_s': 3.35}
- **comm_vs_compute**: {'communication_ratio': 0.0, 'compute_time_seconds': 0.0}

## Recommendations

- Tool execution dominates wall time: optimize sandbox/tool latency or reduce tool calls.
- Memory utilization <30%: increase batch size or use larger context to improve throughput.
