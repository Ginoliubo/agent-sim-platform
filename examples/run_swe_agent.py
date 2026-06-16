"""Example: run a single SWE-agent simulation."""

from agent_sim_platform.config import OPT_LAYER2
from agent_sim_platform.data_models import AgentHarnessSpec, SimulationConfig
from agent_sim_platform.hardware import DEFAULT_REGISTRY as HW_REGISTRY
from agent_sim_platform.models import DEFAULT_REGISTRY as MODEL_REGISTRY
from agent_sim_platform.simulation import run_simulation
from agent_sim_platform.workloads import DEFAULT_REGISTRY as WORKLOAD_REGISTRY


def main():
    config = SimulationConfig(
        hardware=HW_REGISTRY.get("H100-SXM5"),
        model=MODEL_REGISTRY.get("1T-MoE"),
        workload=WORKLOAD_REGISTRY.get("swe-agent"),
        harness=AgentHarnessSpec(name="default"),
        target_context_tokens=32768,
        precision="FP8",
        kv_precision="FP8",
        optimization=OPT_LAYER2,
        random_seed=42,
    )
    result = run_simulation(config)
    print(f"Feasible: {result.feasible}")
    print(f"GPUs: {result.gpu_count}")
    print(f"Latency: {result.latency_seconds:.1f} s")
    print(f"Cost: ${result.cost_usd:.2f}")
    print(f"Bottleneck: {result.bottleneck}")


if __name__ == "__main__":
    main()
