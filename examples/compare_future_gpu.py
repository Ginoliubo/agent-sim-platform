"""Example: compare future NVIDIA and Huawei accelerators."""

from agent_sim_platform.reports import to_markdown
from agent_sim_platform.simulation import sweep_from_names


def main():
    results = sweep_from_names(
        hardware_names=["Rubin", "Rubin-Ultra", "Feynman", "Ascend-950", "Ascend-960", "Ascend-970"],
        model_names=["10T-MoE"],
        workload_name="swe-agent",
        context_tokens=[131072, 1048576],
        optimization_names=["layer2"],
    )
    print(to_markdown(results, title="Future Accelerators: 10T-MoE"))


if __name__ == "__main__":
    main()
