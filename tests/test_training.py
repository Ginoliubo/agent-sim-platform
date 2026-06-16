"""Tests for training simulation engine."""

from agent_sim_platform.data_models import ParallelismConfig, TrainingConfig
from agent_sim_platform.hardware import DEFAULT_REGISTRY as HW_REGISTRY
from agent_sim_platform.models import DEFAULT_REGISTRY as MODEL_REGISTRY
from agent_sim_platform.simulation.training import TrainingEngine, run_training


def test_training_engine_70b():
    cfg = TrainingConfig(
        strategy="pretrain",
        dataset_tokens=1_000_000_000_000,
        epochs=1,
        global_batch_size=4096,
        sequence_length=4096,
        parallelism=ParallelismConfig(dp=64, tp=8, pp=4),
        mfu_target=0.35,
    )
    result = run_training(MODEL_REGISTRY.get("70B-Dense"), HW_REGISTRY.get("H100-SXM5"), cfg)
    assert result.feasible
    assert result.gpu_count > 0
    assert result.latency_seconds > 0
    assert result.metadata["total_flops"] > 0


def test_training_memory_scaling():
    cfg_dp1 = TrainingConfig(
        strategy="pretrain",
        dataset_tokens=10_000_000_000,
        global_batch_size=1024,
        sequence_length=4096,
        parallelism=ParallelismConfig(dp=1, tp=1, pp=1),
        zero_stage=3,
    )
    cfg_dp8 = TrainingConfig(
        strategy="pretrain",
        dataset_tokens=10_000_000_000,
        global_batch_size=1024,
        sequence_length=4096,
        parallelism=ParallelismConfig(dp=8, tp=1, pp=1),
        zero_stage=3,
    )
    model = MODEL_REGISTRY.get("70B-Dense")
    hw = HW_REGISTRY.get("H100-SXM5")
    mem_dp1 = TrainingEngine(model, hw, cfg_dp1)._memory_per_gpu_gb(1)
    mem_dp8 = TrainingEngine(model, hw, cfg_dp8)._memory_per_gpu_gb(8)
    assert mem_dp8 < mem_dp1


def test_training_moe():
    cfg = TrainingConfig(
        strategy="pretrain",
        dataset_tokens=10_000_000_000,
        epochs=1,
        global_batch_size=1024,
        sequence_length=4096,
        parallelism=ParallelismConfig(dp=16, tp=8, pp=2, ep=8),
        mfu_target=0.30,
    )
    result = run_training(MODEL_REGISTRY.get("10T-MoE"), HW_REGISTRY.get("H100-SXM5"), cfg)
    assert result.gpu_count > 0
    assert result.latency_seconds > 0


def test_training_infeasible_without_enough_gpus():
    cfg = TrainingConfig(
        strategy="pretrain",
        dataset_tokens=1_000_000_000,
        epochs=1,
        global_batch_size=1,
        sequence_length=4096,
        parallelism=ParallelismConfig(dp=1, tp=1, pp=1),
    )
    result = run_training(MODEL_REGISTRY.get("10T-MoE"), HW_REGISTRY.get("H100-SXM5"), cfg)
    assert not result.feasible
