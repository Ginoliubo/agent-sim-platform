"""KV-cache offload tier presets.

Models the hierarchical memory stack used in modern LLM inference:
- GPU HBM (hot)
- CPU DRAM (warm)
- Local SSD/NVMe (cold)
- Remote/CXL/ICMS pool (archive / shared)

Bandwidths and latencies are representative 2025-2026 numbers.
"""

from ..data_models import KVOffloadTier

# ---------------------------------------------------------------------------
# Individual tiers
# ---------------------------------------------------------------------------

GPU_HBM_TIER = KVOffloadTier(
    name="gpu_hbm",
    capacity_gb=192.0,
    bandwidth_gb_s=8000.0,
    latency_us=0.1,
    notes="GPU HBM: highest bandwidth, smallest capacity.",
)

CPU_DRAM_TIER = KVOffloadTier(
    name="cpu_dram",
    capacity_gb=2048.0,
    bandwidth_gb_s=64.0,
    latency_us=5.0,
    notes="Host CPU DRAM via PCIe: large capacity, moderate bandwidth.",
)

LOCAL_SSD_TIER = KVOffloadTier(
    name="local_ssd",
    capacity_gb=32768.0,
    bandwidth_gb_s=8.0,
    latency_us=100.0,
    notes="Local NVMe SSD: high capacity, high latency.",
)

REMOTE_DRAM_TIER = KVOffloadTier(
    name="remote_dram",
    capacity_gb=131072.0,
    bandwidth_gb_s=50.0,
    latency_us=20.0,
    notes="Remote DRAM pool via RDMA (e.g. Mooncake CPU pool).",
)

ICMS_TIER = KVOffloadTier(
    name="icms",
    capacity_gb=1048576.0,
    bandwidth_gb_s=25.0,
    latency_us=50.0,
    notes="NVIDIA ICMS/ICMSP: DPU-accelerated RDMA flash pool, pod-scale.",
)

CXL_TIER = KVOffloadTier(
    name="cxl",
    capacity_gb=524288.0,
    bandwidth_gb_s=40.0,
    latency_us=2.0,
    notes="CXL shared memory pool: memory-semantic access at rack scale.",
)

# ---------------------------------------------------------------------------
# Common preset configurations
# ---------------------------------------------------------------------------

NO_OFFLOAD = []

HBM_ONLY = [GPU_HBM_TIER]

HBM_DRAM = [
    GPU_HBM_TIER,
    CPU_DRAM_TIER,
]

HBM_DRAM_SSD = [
    GPU_HBM_TIER,
    CPU_DRAM_TIER,
    LOCAL_SSD_TIER,
]

HBM_ICMS = [
    GPU_HBM_TIER,
    ICMS_TIER,
]

HBM_CXL = [
    GPU_HBM_TIER,
    CXL_TIER,
]

MOONCAKE_LIKE = [
    GPU_HBM_TIER,
    CPU_DRAM_TIER,
    REMOTE_DRAM_TIER,
]

LMCACHE_LIKE = [
    GPU_HBM_TIER,
    CPU_DRAM_TIER,
    LOCAL_SSD_TIER,
]

__all__ = [
    "GPU_HBM_TIER",
    "CPU_DRAM_TIER",
    "LOCAL_SSD_TIER",
    "REMOTE_DRAM_TIER",
    "ICMS_TIER",
    "CXL_TIER",
    "NO_OFFLOAD",
    "HBM_ONLY",
    "HBM_DRAM",
    "HBM_DRAM_SSD",
    "HBM_ICMS",
    "HBM_CXL",
    "MOONCAKE_LIKE",
    "LMCACHE_LIKE",
]
