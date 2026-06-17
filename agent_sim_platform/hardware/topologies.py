"""Network topology presets and registry.

Provides common cluster interconnect patterns used in AI training and inference:
- Single-node NVLink domains
- Fat-Tree InfiniBand/Ethernet clusters
- Rail-optimized designs (common in large LLM training clusters)
"""

from ..data_models import ClusterSpec, NetworkTopology


# ---------------------------------------------------------------------------
# Network topologies
# ---------------------------------------------------------------------------

NVLINK_DOMAIN_8 = NetworkTopology(
    name="nvlink-domain-8",
    topology_type="nvlink-domain",
    gpus_per_node=8,
    intra_node_bw_gb_s=900.0,
    inter_node_bw_gb_s=0.0,
    nics_per_node=0,
    switch_latency_us=0.0,
    oversubscription_ratio=1.0,
    notes="Single-node 8-GPU NVLink domain (e.g. H100-SXM5 DGX baseboard).",
)

NVLINK_DOMAIN_16 = NetworkTopology(
    name="nvlink-domain-16",
    topology_type="nvlink-domain",
    gpus_per_node=16,
    intra_node_bw_gb_s=900.0,
    inter_node_bw_gb_s=0.0,
    nics_per_node=0,
    switch_latency_us=0.0,
    oversubscription_ratio=1.0,
    notes="Single-node 16-GPU NVLink domain (e.g. B200 DGX reference).",
)

FAT_TREE_2_1 = NetworkTopology(
    name="fat-tree-2-1",
    topology_type="fat-tree",
    gpus_per_node=8,
    intra_node_bw_gb_s=900.0,
    inter_node_bw_gb_s=50.0,
    nics_per_node=8,
    switch_latency_us=2.0,
    oversubscription_ratio=2.0,
    notes="Fat-Tree with 2:1 oversubscription, 8x 400 Gbps NICs per node (~50 GB/s each).",
)

FAT_TREE_1_1 = NetworkTopology(
    name="fat-tree-1-1",
    topology_type="fat-tree",
    gpus_per_node=8,
    intra_node_bw_gb_s=900.0,
    inter_node_bw_gb_s=50.0,
    nics_per_node=8,
    switch_latency_us=2.0,
    oversubscription_ratio=1.0,
    notes="Non-blocking Fat-Tree, 8x 400 Gbps NICs per node.",
)

RAIL_OPTIMIZED = NetworkTopology(
    name="rail-optimized",
    topology_type="rail-optimized",
    gpus_per_node=8,
    intra_node_bw_gb_s=900.0,
    inter_node_bw_gb_s=50.0,
    nics_per_node=8,
    switch_latency_us=1.5,
    oversubscription_ratio=1.0,
    notes="Rail-optimized: each GPU NIC lands on its own leaf switch, minimizing all-to-all contention.",
)

# ---------------------------------------------------------------------------
# Topology presets list
# ---------------------------------------------------------------------------

TOPOLOGY_PRESETS = [
    NVLINK_DOMAIN_8,
    NVLINK_DOMAIN_16,
    FAT_TREE_1_1,
    FAT_TREE_2_1,
    RAIL_OPTIMIZED,
]


class TopologyRegistry:
    """Registry of network topology presets."""

    def __init__(self, topologies):
        self._topologies = {t.name: t for t in topologies}

    def get(self, name: str) -> NetworkTopology:
        if name not in self._topologies:
            raise KeyError(f"Unknown topology: {name}")
        return self._topologies[name]

    def list(self):
        return list(self._topologies.values())


DEFAULT_TOPOLOGY_REGISTRY = TopologyRegistry(TOPOLOGY_PRESETS)


# ---------------------------------------------------------------------------
# Cluster presets
# ---------------------------------------------------------------------------

CLUSTER_PRESETS = [
    ClusterSpec(
        name="dgx-h100-8",
        topology=NVLINK_DOMAIN_8,
        node_count=1,
    ),
    ClusterSpec(
        name="fat-tree-256-h100",
        topology=FAT_TREE_2_1,
        node_count=256,
    ),
    ClusterSpec(
        name="rail-256-h100",
        topology=RAIL_OPTIMIZED,
        node_count=256,
    ),
    ClusterSpec(
        name="fat-tree-1024-h100",
        topology=FAT_TREE_1_1,
        node_count=1024,
    ),
    ClusterSpec(
        name="icms-pod-1152-b200",
        topology=RAIL_OPTIMIZED,
        node_count=1152,
    ),
]


class ClusterRegistry:
    """Registry of cluster presets."""

    def __init__(self, clusters):
        self._clusters = {c.name: c for c in clusters}

    def get(self, name: str) -> ClusterSpec:
        if name not in self._clusters:
            raise KeyError(f"Unknown cluster: {name}")
        return self._clusters[name]

    def list(self):
        return list(self._clusters.values())


DEFAULT_CLUSTER_REGISTRY = ClusterRegistry(CLUSTER_PRESETS)


__all__ = [
    "NetworkTopology",
    "ClusterSpec",
    "TopologyRegistry",
    "ClusterRegistry",
    "DEFAULT_TOPOLOGY_REGISTRY",
    "DEFAULT_CLUSTER_REGISTRY",
    "TOPOLOGY_PRESETS",
    "CLUSTER_PRESETS",
    "NVLINK_DOMAIN_8",
    "NVLINK_DOMAIN_16",
    "FAT_TREE_1_1",
    "FAT_TREE_2_1",
    "RAIL_OPTIMIZED",
]
