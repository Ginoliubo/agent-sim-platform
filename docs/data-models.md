# Data Models

## Core Dataclasses

### `HardwareSpec`

Immutable accelerator specification.

```python
@dataclass(frozen=True)
class HardwareSpec:
    name: str
    vendor: str          # nvidia / huawei / google
    kind: str            # gpu / npu / tpu
    memory_gb: float
    memory_bw_tb_s: float
    fp16_tflops: float
    fp8_tflops: float
    fp4_tflops: float = 0.0
    interconnect_bw_gb_s: float = 0.0
    pcie_bw_gb_s: float = 0.0
    power_w: float = 0.0
    cost_per_hour: float = 0.0
    release_year: int
    is_future: bool = False
    uncertainty_range: float = 0.0
```

### `ModelSpec`

Immutable model specification with helper methods for memory and KV size.

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str
    total_params_b: float
    active_params_b: float
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    num_experts: int = 1
    top_k: int = 1
    architecture: str = "dense"   # dense | moe
```

Key methods:
- `weight_memory_gb(precision)`
- `active_weight_memory_gb(precision)`
- `kv_bytes_per_token(kv_precision)`
- `gpu_needed(memory_gb, precision, overhead)`

### `WorkloadSpec`

Stochastic agent workload.

```python
@dataclass
class WorkloadSpec:
    name: str
    max_steps: int
    avg_steps: float
    step_std: float
    context_limit: int
    token_distributions: Dict[str, Tuple[float, float]]
    tool_delays: Dict[str, float]
    tool_probs: Dict[str, float]
    env_init_delay: float
```

### `AgentHarnessSpec`

Agent software harness configuration.

```python
@dataclass
class AgentHarnessSpec:
    name: str
    concurrency: int
    control_cpu_percent: float
    agent_cpu_peak_cores: int
    compile_test_cores: int
    sandbox_type: str
    compaction_strategy: Optional[str]
    sub_agent_enabled: bool
    sub_agent_count: int
```

### `OptimizationConfig`

Stack of inference optimizations.

```python
@dataclass
class OptimizationConfig:
    name: str
    prefix_caching: bool
    prefix_caching_hit_rate: float
    flash_attention_speedup: float
    spec_decode_speedup: float
    continuous_batching_efficiency: float
    kv_compression_ratio: float
    hbf_offload: bool
    hbf_offload_ratio: float
    hbf_bw_gb_s: float
    quantization_bits: int
```

Presets: `baseline`, `layer1`, `layer2`, `layer3`.

### `SimulationConfig`

Complete input for a run.

```python
@dataclass
class SimulationConfig:
    hardware: HardwareSpec
    model: ModelSpec
    workload: WorkloadSpec
    harness: AgentHarnessSpec
    target_context_tokens: int
    precision: str
    kv_precision: str
    tp: int
    pp: int
    batch_size: int
    optimization: OptimizationConfig
    random_seed: int
```

### `SimulationResult`

Unified output.

```python
@dataclass
class SimulationResult:
    config: SimulationConfig
    latency_seconds: float
    wall_time_seconds: float
    tokens_total: int
    peak_kv_gb: float
    memory_required_gb: float
    gpu_count: int
    feasible: bool
    bottleneck: str
    cost_usd: float
    utilization_gpu: float
    per_step: List[Dict]
    metadata: Dict
```

## Units

- Memory: GB (decimal, 1 GB = 1e9 bytes)
- Bandwidth: TB/s (decimal)
- Compute: TFLOPS
- Time: seconds
- Cost: USD
