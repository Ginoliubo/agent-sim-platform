"""Example: sweep hardware and context for a fixed model."""

from agent_sim_platform.reports import to_markdown
from agent_sim_platform.simulation import sweep_from_names


def main():
    results = sweep_from_names(
        hardware_names=["H100-SXM5", "B200", "Rubin"],
        model_names=["10T-MoE"],
        workload_name="swe-agent",
        context_tokens=[32768, 131072, 1048576],
        optimization_names=["baseline", "layer2"],
    )
    print(to_markdown(results, title="10T-MoE Hardware Sweep"))


if __name__ == "__main__":
    main()
