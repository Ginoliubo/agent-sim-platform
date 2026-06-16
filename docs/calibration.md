# Calibration

## Overview

The calibration engine compares simulation outputs against benchmark fixtures and reports the error. It can also **auto-fit** a selected set of hard-coded constants to minimize the overall MAPE.

## Supported Metrics

- **MAPE** (Mean Absolute Percentage Error): primary calibration metric
- **RMSE** (Root Mean Squared Error)
- **R²** (coefficient of determination)

## Tunable Constants

| Constant | Affects | Typical Range |
|----------|---------|---------------|
| `mfu_target` | Training compute time | 0.10 – 0.70 |
| `default_prefill_utilization` | Capacity / serving prefill | 0.10 – 0.70 |
| `default_decode_utilization` | Capacity / serving decode | 0.10 – 0.90 |
| `activation_overhead_factor` | Training memory | 0.5 – 2.0 |
| `continuous_batching_efficiency` | Serving throughput | 0.5 – 2.0 |
| `kv_compression_ratio` | KV-cache memory | 0.1 – 1.0 |

## CLI

```bash
# Evaluate a single fixture
agent-sim benchmark --name llama2_70b_pretrain

# Calibrate all domains and print a Markdown report
agent-sim calibrate --domain all

# Calibrate only training, fitting selected constants
agent-sim calibrate \
  --domain training \
  --fit-params mfu_target,activation_overhead_factor \
  --max-iterations 50 \
  --output calibration.json

# Calibrate serving latency constants
agent-sim calibrate \
  --domain serving \
  --fit-params default_decode_utilization,continuous_batching_efficiency
```

## Programmatic Use

```python
from agent_sim_platform.benchmarks import DEFAULT_REGISTRY
from agent_sim_platform.calibration import CalibrationConfig, CalibrationEngine

engine = CalibrationEngine(CalibrationConfig(domain="training"))
report = engine.fit(DEFAULT_REGISTRY)

print(f"Overall MAPE: {report.overall_mape:.2%}")
for name, value in report.fitted_values.items():
    print(f"  {name} = {value:.4f}")
```

## Interpretation

- `overall_mape < 20%`: simulation is well-calibrated for the fixture set.
- `20% < overall_mape < 40%`: acceptable for early-stage capacity planning; inspect per-fixture errors.
- `overall_mape > 40%`: fixture assumptions or simulation model may need revision.

## Caveats

- Public benchmarks often omit exact parallelism, optimization flags, or cluster topology. Fixtures explicitly document inferred values in `notes`.
- Auto-fit uses coordinate descent over a bounded grid; it improves the fit but cannot fix fundamental modeling gaps.
- Future hardware (`is_future=True`) is excluded from calibration by default.
