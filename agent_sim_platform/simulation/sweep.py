"""Parameter sweep across hardware, models, contexts, and optimizations."""

from itertools import product
from typing import Dict, Iterable, List

from ..data_models import (
    AgentHarnessSpec,
    HardwareSpec,
    ModelSpec,
    OptimizationConfig,
    SimulationConfig,
    SimulationResult,
    WorkloadSpec,
)
from ..hardware import HardwareRegistry
from ..models import ModelRegistry
from ..workloads import WorkloadRegistry
from .engine import SimulationEngine


def sweep(
    hardware_specs: Iterable[HardwareSpec],
    model_specs: Iterable[ModelSpec],
    workload_spec: WorkloadSpec,
    harness_spec: AgentHarnessSpec,
    context_tokens: Iterable[int],
    optimizations: Iterable[OptimizationConfig],
    precision: str = "FP8",
    kv_precision: str = "FP8",
    seed: int = 42,
) -> List[SimulationResult]:
    """Grid sweep over hardware × model × context × optimization.

    Returns one SimulationResult per configuration combination.
    """
    results = []
    for hw, model, ctx, opt in product(
        hardware_specs, model_specs, context_tokens, optimizations
    ):
        config = SimulationConfig(
            hardware=hw,
            model=model,
            workload=workload_spec,
            harness=harness_spec,
            target_context_tokens=ctx,
            precision=precision,
            kv_precision=kv_precision,
            optimization=opt,
            random_seed=seed,
        )
        result = SimulationEngine(config).run()
        results.append(result)
    return results


def sweep_from_names(
    hardware_names: List[str],
    model_names: List[str],
    workload_name: str,
    context_tokens: List[int],
    optimization_names: List[str],
    harness_name: str = "default",
    precision: str = "FP8",
    kv_precision: str = "FP8",
    seed: int = 42,
    hardware_registry: HardwareRegistry = None,
    model_registry: ModelRegistry = None,
    workload_registry: WorkloadRegistry = None,
) -> List[SimulationResult]:
    """Convenience sweep using registered preset names."""
    from ..config import OPTIMIZATION_PRESETS
    from ..hardware import DEFAULT_REGISTRY as HW_REGISTRY
    from ..models import DEFAULT_REGISTRY as MODEL_REGISTRY
    from ..workloads import DEFAULT_REGISTRY as WORKLOAD_REGISTRY

    hw_reg = hardware_registry or HW_REGISTRY
    model_reg = model_registry or MODEL_REGISTRY
    wl_reg = workload_registry or WORKLOAD_REGISTRY

    hardware_specs = [hw_reg.get(name) for name in hardware_names]
    model_specs = [model_reg.get(name) for name in model_names]
    workload_spec = wl_reg.get(workload_name)
    optimizations = [OPTIMIZATION_PRESETS[name] for name in optimization_names]
    harness_spec = AgentHarnessSpec(name=harness_name)

    return sweep(
        hardware_specs=hardware_specs,
        model_specs=model_specs,
        workload_spec=workload_spec,
        harness_spec=harness_spec,
        context_tokens=context_tokens,
        optimizations=optimizations,
        precision=precision,
        kv_precision=kv_precision,
        seed=seed,
    )


def sweep_to_dict(results: List[SimulationResult]) -> List[Dict]:
    """Convert sweep results to a list of plain dictionaries."""
    return [r.to_dict() for r in results]
