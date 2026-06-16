"""Training job workload specification."""

from dataclasses import dataclass, field

from ..data_models import ParallelismConfig, TrainingConfig


@dataclass
class TrainingWorkload:
    """A training workload definition."""

    name: str
    training_config: TrainingConfig
    description: str = ""


# Common presets
PRETRAIN_WORKLOAD = TrainingWorkload(
    name="pretrain",
    training_config=TrainingConfig(
        strategy="pretrain",
        global_batch_size=4096,
        micro_batch_size=1,
        sequence_length=4096,
        gradient_checkpointing=True,
        zero_stage=1,
        mfu_target=0.35,
    ),
    description="Standard pretraining workload.",
)

SFT_WORKLOAD = TrainingWorkload(
    name="sft",
    training_config=TrainingConfig(
        strategy="sft",
        global_batch_size=512,
        micro_batch_size=1,
        sequence_length=4096,
        gradient_checkpointing=True,
        zero_stage=1,
        mfu_target=0.30,
    ),
    description="Supervised fine-tuning workload.",
)

RLHF_WORKLOAD = TrainingWorkload(
    name="rlhf",
    training_config=TrainingConfig(
        strategy="rlhf",
        global_batch_size=256,
        micro_batch_size=1,
        sequence_length=2048,
        gradient_checkpointing=True,
        zero_stage=1,
        mfu_target=0.25,
    ),
    description="RLHF/PPO workload.",
)

GRPO_WORKLOAD = TrainingWorkload(
    name="grpo",
    training_config=TrainingConfig(
        strategy="grpo",
        global_batch_size=256,
        micro_batch_size=1,
        sequence_length=2048,
        gradient_checkpointing=True,
        zero_stage=1,
        mfu_target=0.25,
    ),
    description="Group Relative Policy Optimization workload.",
)

TRAINING_WORKLOAD_PRESETS = [PRETRAIN_WORKLOAD, SFT_WORKLOAD, RLHF_WORKLOAD, GRPO_WORKLOAD]


class TrainingWorkloadRegistry:
    """Registry for training workload presets."""

    def __init__(self, workloads=None):
        self._workloads = {}
        if workloads:
            for w in workloads:
                self.register(w)

    def register(self, workload: TrainingWorkload) -> None:
        if workload.name in self._workloads:
            raise ValueError(f"Training workload '{workload.name}' already registered")
        self._workloads[workload.name] = workload

    def get(self, name: str) -> TrainingWorkload:
        if name not in self._workloads:
            raise KeyError(f"Training workload '{name}' not found")
        return self._workloads[name]

    def list(self):
        return sorted(self._workloads.values(), key=lambda w: w.name)

    def names(self):
        return sorted(self._workloads.keys())


DEFAULT_REGISTRY = TrainingWorkloadRegistry(TRAINING_WORKLOAD_PRESETS)

__all__ = [
    "TrainingWorkload",
    "PRETRAIN_WORKLOAD",
    "SFT_WORKLOAD",
    "RLHF_WORKLOAD",
    "GRPO_WORKLOAD",
    "TRAINING_WORKLOAD_PRESETS",
    "TrainingWorkloadRegistry",
    "DEFAULT_REGISTRY",
]
