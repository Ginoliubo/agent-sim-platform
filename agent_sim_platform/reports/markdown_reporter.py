"""Markdown reporter for simulation results."""

from typing import List

from ..data_models import SimulationResult
from ..utils.units import format_bytes, seconds_to_hms


def _fmt_bool(value: bool) -> str:
    return "✅ Yes" if value else "❌ No"


def _single_result_md(result: SimulationResult) -> str:
    cfg = result.config
    lines = [
        "## Simulation Result",
        "",
        "### Configuration",
        f"- **Hardware**: {cfg.hardware.name} ({cfg.hardware.vendor} {cfg.hardware.kind})",
        f"- **Model**: {cfg.model.name} ({cfg.model.architecture}, {cfg.model.total_params_b:.0f}B total / {cfg.model.active_params_b:.0f}B active)",
        f"- **Workload**: {cfg.workload.name}",
        f"- **Context**: {cfg.target_context_tokens:,} tokens",
        f"- **Precision**: {cfg.precision} / KV {cfg.kv_precision}",
        f"- **Optimization**: {cfg.optimization.name}",
        f"- **Parallelism**: TP={cfg.tp}, PP={cfg.pp}, BS={cfg.batch_size}",
        "",
        "### Summary",
        f"- **Feasible**: {_fmt_bool(result.feasible)}",
        f"- **Bottleneck**: {result.bottleneck}",
        f"- **GPU Count**: {result.gpu_count}",
        f"- **Latency**: {seconds_to_hms(result.latency_seconds)} ({result.latency_seconds:.2f} s)",
        f"- **Wall Time (concurrency={cfg.harness.concurrency})**: {seconds_to_hms(result.wall_time_seconds)}",
        f"- **Total Tokens**: {result.tokens_total:,} (input {result.tokens_input:,}, output {result.tokens_output:,})",
        f"- **Peak KV Cache**: {format_bytes(result.peak_kv_gb * 1e9)}",
        f"- **Memory Required**: {format_bytes(result.memory_required_gb * 1e9)}",
        f"- **GPU Utilization**: {result.utilization_gpu:.1%}",
        f"- **Cost**: ${result.cost_usd:.2f}",
        "",
    ]
    if result.metadata:
        lines.append("### Metadata")
        for key, value in result.metadata.items():
            if isinstance(value, float):
                lines.append(f"- **{key}**: {value:.4f}")
            else:
                lines.append(f"- **{key}**: {value}")
        lines.append("")
    return "\n".join(lines)


def _comparison_table(results: List[SimulationResult]) -> str:
    lines = [
        "## Comparison Table",
        "",
        "| Hardware | Model | Context | Feasible | GPUs | Latency | Tokens | Peak KV | Cost | Bottleneck |",
        "|----------|-------|---------|----------|------|---------|--------|---------|------|------------|",
    ]
    for r in results:
        lines.append(
            f"| {r.config.hardware.name} | {r.config.model.name} | "
            f"{r.config.target_context_tokens:,} | {r.feasible} | {r.gpu_count} | "
            f"{seconds_to_hms(r.latency_seconds)} | {r.tokens_total:,} | "
            f"{format_bytes(r.peak_kv_gb * 1e9)} | ${r.cost_usd:.2f} | {r.bottleneck} |"
        )
    lines.append("")
    return "\n".join(lines)


def to_markdown(results: List[SimulationResult], title: str = "Agent Simulation Report") -> str:
    """Generate a Markdown report from one or more SimulationResults."""
    lines = [f"# {title}", ""]
    if len(results) == 1:
        lines.append(_single_result_md(results[0]))
    else:
        lines.append(_comparison_table(results))
        lines.append("")
        for idx, result in enumerate(results, 1):
            lines.append(f"### Config {idx}: {result.config.hardware.name} × {result.config.model.name}")
            lines.append("")
            lines.append(_single_result_md(result))
    return "\n".join(lines)
