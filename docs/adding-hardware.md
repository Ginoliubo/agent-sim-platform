# Adding Hardware

## Step 1: Create or edit a vendor preset file

Vendor files live in `agent_sim_platform/hardware/`:

- `nvidia.py`
- `huawei.py`
- `google.py`

## Step 2: Define a `HardwareSpec`

```python
from .base import HardwareSpec

MY_GPU = HardwareSpec(
    name="MyGPU",
    vendor="myvendor",
    kind="gpu",
    memory_gb=192.0,
    memory_bw_tb_s=8.0,
    fp16_tflops=4500.0,
    fp8_tflops=9000.0,
    fp4_tflops=18000.0,
    interconnect_bw_gb_s=1800.0,
    pcie_bw_gb_s=128.0,
    power_w=1000.0,
    cost_per_hour=6.0,
    release_year=2026,
    is_future=False,
    uncertainty_range=0.0,
    notes="Optional notes",
)
```

For future/unreleased hardware, set `is_future=True` and provide an `uncertainty_range` (e.g., `0.25` for ±25%).

## Step 3: Add to the vendor list

```python
MY_HARDWARE = [
    # ... existing specs ...
    MY_GPU,
]
```

## Step 4: Verify

```bash
python3 -m agent_sim_platform list-hardware
```

Your new hardware should appear in the list.

## Step 5: Use it

```bash
python3 -m agent_sim_platform run \
  --hardware MyGPU \
  --model 1T-MoE \
  --context 32K
```

## Tips

- Keep `fp4_tflops=0.0` if the accelerator does not support FP4.
- `interconnect_bw_gb_s` is the aggregate per-node bandwidth (NVLink, HCCS, ICI, etc.).
- `cost_per_hour` is used for rental cost estimation and can be left as `0.0` if unknown.
