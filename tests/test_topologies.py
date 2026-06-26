"""Tests for network topology, cluster, and distributed inference modeling."""

import pytest

from agent_sim_platform.config import OPTIMIZATION_PRESETS
from agent_sim_platform.data_models import (
    AFDConfig,
    ClusterSpec,
    HardwareSpec,
    InferenceServiceConfig,
    KVOffloadConfig,
    NetworkTopology,
    OptimizationConfig,
    PDConfig,
    ParallelismConfig,
    TrainingConfig,
)
from agent_sim_platform.hardware import (
    DEFAULT_CLUSTER_REGISTRY,
    DEFAULT_TOPOLOGY_REGISTRY,
    HBM_DRAM_SSD,
)
from agent_sim_platform.hardware import DEFAULT_REGISTRY as HW_REGISTRY
from agent_sim_platform.models import DEFAULT_REGISTRY as MODEL_REGISTRY
from agent_sim_platform.simulation.cluster_capacity import ClusterCapacityEstimator
from agent_sim_platform.simulation.inference_serving import InferenceServingEngine
from agent_sim_platform.simulation.training import TrainingEngine


def test_topology_registry_lists_presets():
    topologies = DEFAULT_TOPOLOGY_REGISTRY.list()
    names = {t.name for t in topologies}
    assert "nvlink-domain-8" in names
    assert "fat-tree-1-1" in names
    assert "rail-optimized" in names


def test_topology_effective_inter_node_bandwidth():
    ft = DEFAULT_TOPOLOGY_REGISTRY.get("fat-tree-2-1")
    # 8 NICs * 50 GB/s / 2:1 oversubscription
    assert ft.aggregate_inter_node_bw_gb_s == 400.0
    assert ft.effective_inter_node_bw_gb_s == 200.0


def test_cluster_registry_lists_presets():
    clusters = DEFAULT_CLUSTER_REGISTRY.list()
    names = {c.name for c in clusters}
    assert "dgx-h100-8" in names
    assert "fat-tree-256-h100" in names


def test_cluster_total_gpus():
    cluster = DEFAULT_CLUSTER_REGISTRY.get("fat-tree-256-h100")
    assert cluster.total_gpus == 2048
    assert cluster.gpus_per_node == 8


def test_distributed_inference_memory_per_gpu():
    model = MODEL_REGISTRY.get("1T-Dense")
    hw = HW_REGISTRY.get("B200")
    cluster = DEFAULT_CLUSTER_REGISTRY.get("fat-tree-1024-h100")
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=1.0,
        request_length_mean=10000000,
        output_length_mean=1000,
        simulation_duration_seconds=1.0,
        max_batch_size=1,
    )
    engine = InferenceServingEngine(
        model=model,
        hardware=hw,
        service_config=cfg,
        precision="FP8",
        kv_precision="FP8",
        cluster=cluster,
        tp=8,
        pp=4,
        cp=64,
        optimization=OPTIMIZATION_PRESETS["layer3"],
    )
    mem_per_gpu = engine._memory_per_gpu_gb(10000000, batch_size=1)
    # Should be well under 192 GB with this sharding
    assert mem_per_gpu < hw.memory_gb
    assert engine.gpu_count == 8 * 4 * 64


def test_distributed_inference_infeasible_without_enough_sharding():
    model = MODEL_REGISTRY.get("1T-Dense")
    hw = HW_REGISTRY.get("B200")
    cluster = DEFAULT_CLUSTER_REGISTRY.get("fat-tree-256-h100")
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=1.0,
        request_length_mean=10000000,
        output_length_mean=1000,
        simulation_duration_seconds=1.0,
        max_batch_size=1,
    )
    engine = InferenceServingEngine(
        model=model,
        hardware=hw,
        service_config=cfg,
        precision="FP8",
        kv_precision="FP8",
        cluster=cluster,
        tp=8,
        pp=1,
        cp=32,
        optimization=OPTIMIZATION_PRESETS["layer3"],
    )
    assert not engine._fits_in_cluster(10000000, batch_size=1)


def test_distributed_inference_communication_breakdown():
    model = MODEL_REGISTRY.get("1T-Dense")
    hw = HW_REGISTRY.get("B200")
    cluster = DEFAULT_CLUSTER_REGISTRY.get("fat-tree-1024-h100")
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=1.0,
        request_length_mean=10000000,
        output_length_mean=1000,
        simulation_duration_seconds=1.0,
        max_batch_size=1,
    )
    engine = InferenceServingEngine(
        model=model,
        hardware=hw,
        service_config=cfg,
        precision="FP8",
        kv_precision="FP8",
        cluster=cluster,
        tp=8,
        pp=4,
        cp=64,
        optimization=OPTIMIZATION_PRESETS["layer3"],
    )
    comm = engine._communication_breakdown(engine.service_config.request_length_mean, 1)
    # CP ring communication is present and positive for long-context inference
    assert comm.cp_bytes_per_token > 0
    assert comm.cp_time_per_token_ms > 0


def test_cluster_capacity_finds_feasible_config():
    model = MODEL_REGISTRY.get("1T-Dense")
    hw = HW_REGISTRY.get("B200")
    cluster = DEFAULT_CLUSTER_REGISTRY.get("fat-tree-1024-h100")
    estimator = ClusterCapacityEstimator(
        model=model,
        hardware=hw,
        cluster=cluster,
        precision="FP8",
        kv_precision="FP8",
        optimization=OPTIMIZATION_PRESETS["layer3"],
    )
    result = estimator.find_minimal_config(10000000)
    assert result.feasible
    assert result.gpu_count <= cluster.total_gpus
    assert result.memory_per_gpu_gb <= hw.memory_gb


def test_cluster_capacity_exceeds_cluster_size():
    model = MODEL_REGISTRY.get("1T-Dense")
    hw = HW_REGISTRY.get("B200")
    cluster = DEFAULT_CLUSTER_REGISTRY.get("dgx-h100-8")
    estimator = ClusterCapacityEstimator(
        model=model,
        hardware=hw,
        cluster=cluster,
        precision="FP8",
        kv_precision="FP8",
        optimization=OPTIMIZATION_PRESETS["layer3"],
    )
    result = estimator.find_minimal_config(10000000)
    assert not result.feasible


def test_training_topology_aware_communication():
    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("H100-SXM5")
    cluster = DEFAULT_CLUSTER_REGISTRY.get("fat-tree-256-h100")
    cfg = TrainingConfig(
        strategy="pretrain",
        dataset_tokens=1_000_000_000,
        global_batch_size=1024,
        micro_batch_size=1,
        sequence_length=4096,
        parallelism=ParallelismConfig(dp=4, tp=8, pp=2),
    )
    engine = TrainingEngine(model, hw, cfg, precision="FP8", cluster=cluster)
    gpu_count = cfg.parallelism.total_gpus
    compute_time = engine._compute_time_per_step(gpu_count)
    comm_time = engine._communication_time_per_step(gpu_count, compute_time)
    assert comm_time >= 0
    # With 64 GPUs per node and dp=4,tp=8,pp=2 = 64 GPUs total, single node if nodes=8?
    # Actually cluster is 256 nodes * 8 = 2048 GPUs, but we only use 64.
    # The mapping matters: if we use 8 nodes (64 GPUs), dp=4,tp=8,pp=2 fits with TP intra-node.
    assert engine._cross_node_fraction(8) == 0.0  # TP fits in node
    assert engine._cross_node_fraction(4) == 0.0  # DP fits in node if mapped well


def test_training_cross_node_communication_slower():
    """Cross-node traffic should increase communication time vs intra-node only."""
    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("H100-SXM5")
    cfg = TrainingConfig(
        strategy="pretrain",
        dataset_tokens=1_000_000_000,
        global_batch_size=1024,
        micro_batch_size=1,
        sequence_length=4096,
        parallelism=ParallelismConfig(dp=64, tp=1, pp=1),
    )
    # Single-node NVLink: all DP traffic intra-node
    single_node = ClusterSpec(
        name="single-node",
        topology=DEFAULT_TOPOLOGY_REGISTRY.get("nvlink-domain-8"),
        node_count=8,  # 64 GPUs
    )
    # Fat-tree: DP traffic crosses nodes
    multi_node = ClusterSpec(
        name="multi-node",
        topology=DEFAULT_TOPOLOGY_REGISTRY.get("fat-tree-1-1"),
        node_count=8,
    )

    gpu_count = 64
    single = TrainingEngine(model, hw, cfg, precision="FP8", cluster=single_node)
    multi = TrainingEngine(model, hw, cfg, precision="FP8", cluster=multi_node)
    ct = single._compute_time_per_step(gpu_count)
    single_comm = single._communication_time_per_step(gpu_count, ct)
    multi_comm = multi._communication_time_per_step(gpu_count, ct)
    assert multi_comm > single_comm


def test_training_network_utilization_present():
    """Training result should include network utilization and communication breakdown."""
    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("H100-SXM5")
    cluster = DEFAULT_CLUSTER_REGISTRY.get("fat-tree-256-h100")
    cfg = TrainingConfig(
        strategy="pretrain",
        dataset_tokens=1_000_000_000,
        global_batch_size=1024,
        micro_batch_size=1,
        sequence_length=4096,
        parallelism=ParallelismConfig(dp=4, tp=8, pp=2),
    )
    result = TrainingEngine(model, hw, cfg, precision="FP8", cluster=cluster).run()
    assert "network_utilization" in result.metadata
    assert "communication_breakdown_seconds" in result.metadata
    assert 0.0 <= result.metadata["network_utilization"] <= 1.0


def test_training_bottleneck_identifies_network():
    """Cross-node DP with constrained cluster should flag network bottleneck."""
    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("H100-SXM5")
    low_bw_topology = NetworkTopology(
        name="low-bw",
        topology_type="fat-tree",
        gpus_per_node=8,
        intra_node_bw_gb_s=900.0,
        inter_node_bw_gb_s=1.0,
        nics_per_node=1,
        oversubscription_ratio=1.0,
    )
    cluster = ClusterSpec(name="low-bw-cluster", topology=low_bw_topology, node_count=8)
    cfg = TrainingConfig(
        strategy="pretrain",
        dataset_tokens=1_000_000_000,
        global_batch_size=1024,
        micro_batch_size=1,
        sequence_length=4096,
        zero_stage=2,
        parallelism=ParallelismConfig(dp=16, tp=4, pp=1),
    )
    result = TrainingEngine(model, hw, cfg, precision="FP8", cluster=cluster).run()
    assert "network" in result.bottleneck or "cross-node" in result.bottleneck


def test_mla_kv_cache_reduction():
    """MLA should produce much smaller KV cache than dense GQA."""
    from agent_sim_platform.algorithms import DEFAULT_REGISTRY as ALGO_REGISTRY

    mla_algo = ALGO_REGISTRY.get("mla")
    dense_algo = ALGO_REGISTRY.get("dense")
    model = MODEL_REGISTRY.get("1T-MoE-MLA")
    dense_model = MODEL_REGISTRY.get("1T-Dense")

    mla_kv = model.kv_bytes_per_token("FP8")
    dense_kv = dense_model.kv_bytes_per_token("FP8")
    assert mla_kv < dense_kv / 10
    assert model.algorithm_family.name == "mla"


def test_1t_moe_mla_feasible_at_10m_context():
    """1T-MoE-MLA should fit in a modest B200 cluster at 10M context."""
    model = MODEL_REGISTRY.get("1T-MoE-MLA")
    hw = HW_REGISTRY.get("B200")
    cluster = DEFAULT_CLUSTER_REGISTRY.get("fat-tree-1024-h100")
    estimator = ClusterCapacityEstimator(
        model=model,
        hardware=hw,
        cluster=cluster,
        precision="FP8",
        kv_precision="FP8",
        optimization=OPTIMIZATION_PRESETS["layer3"],
    )
    result = estimator.find_minimal_config(10_000_000)
    assert result.feasible
    assert result.gpu_count <= cluster.total_gpus


def test_pd_disaggregation_adds_kv_transfer_time():
    """PD separation should add KV transfer time to TTFT."""
    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("H100-SXM5")
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=1.0,
        request_length_mean=4096,
        output_length_mean=512,
        simulation_duration_seconds=10.0,
        max_batch_size=4,
        pd_config=PDConfig(enabled=True, prefill_gpu_count=4, decode_gpu_count=4, kv_transfer_bw_gb_s=100.0),
    )
    result = InferenceServingEngine(model, hw, cfg, gpu_count=8).run()
    assert result.metadata["pd_enabled"]
    assert result.metadata["kv_transfer_time_per_request_ms"] > 0


def test_kv_offload_tiered_access_adds_latency():
    """Tiered KV offload should add nonzero access time per token."""
    from agent_sim_platform.hardware import HBM_DRAM_SSD
    from agent_sim_platform.data_models import KVOffloadConfig

    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("H100-SXM5")
    tiers = [t for t in HBM_DRAM_SSD]
    # Set hit rates
    for t in tiers:
        if t.name == "gpu_hbm":
            t.hit_rate = 0.5
        elif t.name == "cpu_dram":
            t.hit_rate = 0.4
        elif t.name == "local_ssd":
            t.hit_rate = 0.1
    opt = OPTIMIZATION_PRESETS["baseline"]
    opt = OptimizationConfig(**{**opt.__dict__, "kv_offload": KVOffloadConfig(tiers=tiers)})

    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=1.0,
        request_length_mean=4096,
        output_length_mean=512,
        simulation_duration_seconds=10.0,
        max_batch_size=4,
    )
    result = InferenceServingEngine(model, hw, cfg, gpu_count=8, optimization=opt).run()
    assert result.metadata["kv_offload_time_per_token_ms"] > 0


def test_afd_disaggregation_adds_activation_transfer_time():
    """AFD separation should add activation transfer time to decode."""
    model = MODEL_REGISTRY.get("1T-MoE")
    hw = HW_REGISTRY.get("H100-SXM5")
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=1.0,
        request_length_mean=4096,
        output_length_mean=512,
        simulation_duration_seconds=10.0,
        max_batch_size=2,
        afd_config=AFDConfig(enabled=True, attention_gpu_count=4, ffn_gpu_count=4, decode_gpu_count=4),
    )
    result = InferenceServingEngine(model, hw, cfg, gpu_count=12).run()
    assert result.metadata["afd_enabled"]
    assert result.metadata["afd_transfer_time_per_token_ms"] > 0


def test_serving_bottleneck_identifies_network():
    """Long-context CP-heavy config on a constrained cluster should flag network bottleneck."""
    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("B200")
    low_bw_topology = NetworkTopology(
        name="low-bw",
        topology_type="fat-tree",
        gpus_per_node=8,
        intra_node_bw_gb_s=900.0,
        inter_node_bw_gb_s=0.1,
        nics_per_node=1,
        oversubscription_ratio=1.0,
    )
    cluster = ClusterSpec(name="low-bw-cluster", topology=low_bw_topology, node_count=8)
    opt = OptimizationConfig(**{**OPTIMIZATION_PRESETS["layer3"].__dict__, "decode_overhead_ms": -1.0})
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=1.0,
        request_length_mean=10000000,
        output_length_mean=1000,
        simulation_duration_seconds=1.0,
        max_batch_size=1,
    )
    result = InferenceServingEngine(
        model=model,
        hardware=hw,
        service_config=cfg,
        precision="FP8",
        kv_precision="FP8",
        cluster=cluster,
        tp=8,
        pp=1,
        cp=8,
        optimization=opt,
    ).run()
    assert "network" in result.bottleneck


def test_serving_network_utilization_present():
    """Network utilization metadata should be present and bounded."""
    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("H100-SXM5")
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=1.0,
        request_length_mean=4096,
        output_length_mean=512,
        simulation_duration_seconds=10.0,
        max_batch_size=4,
    )
    result = InferenceServingEngine(model, hw, cfg, gpu_count=8).run()
    assert "network_utilization" in result.metadata
    assert 0.0 <= result.metadata["network_utilization"] <= 1.0


def test_cluster_caps_pd_kv_transfer_bandwidth():
    """Pool-to-pool KV transfer bandwidth should be capped by cluster inter-node bandwidth."""
    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("H100-SXM5")
    low_bw_topology = NetworkTopology(
        name="low-bw-fat-tree",
        topology_type="fat-tree",
        gpus_per_node=8,
        intra_node_bw_gb_s=900.0,
        inter_node_bw_gb_s=10.0,
        nics_per_node=1,
        oversubscription_ratio=1.0,
    )
    cluster = ClusterSpec(name="low-bw-cluster", topology=low_bw_topology, node_count=8)
    cfg = InferenceServiceConfig(
        arrival_rate_per_sec=1.0,
        request_length_mean=4096,
        output_length_mean=512,
        simulation_duration_seconds=10.0,
        max_batch_size=4,
        pd_config=PDConfig(
            enabled=True,
            prefill_gpu_count=4,
            decode_gpu_count=4,
            kv_transfer_bw_gb_s=1000.0,  # would-be fast, should be capped
        ),
    )
    result = InferenceServingEngine(
        model=model,
        hardware=hw,
        service_config=cfg,
        cluster=cluster,
        gpu_count=8,
    ).run()
    # Effective bandwidth = min(1000, 10) GB/s, so transfer should be slow
    assert result.metadata["kv_transfer_time_per_request_ms"] > 1.0
