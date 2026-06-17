"""Unified CLI for agent-sim."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .algorithms import DEFAULT_REGISTRY as ALGORITHM_REGISTRY
from .config import OPTIMIZATION_PRESETS
from .benchmarks import DEFAULT_REGISTRY as BENCHMARK_REGISTRY
from .calibration import CalibrationConfig, CalibrationEngine, format_report
from .data_models import (
    AgentHarnessSpec,
    AFDConfig,
    ClusterSpec,
    InferenceServiceConfig,
    KVOffloadConfig,
    OptimizationConfig,
    PDConfig,
    ParallelismConfig,
    SimulationConfig,
    SimulationResult,
    TrainingConfig,
)
from .hardware import DEFAULT_REGISTRY as HW_REGISTRY
from .hardware import DEFAULT_TOPOLOGY_REGISTRY, DEFAULT_CLUSTER_REGISTRY
from .models import DEFAULT_REGISTRY as MODEL_REGISTRY
from .profiling import ProfilingOrchestrator
from .reports.json_reporter import to_json
from .reports.markdown_reporter import to_markdown
from .simulation import (
    CapacityEstimator,
    ClusterCapacityEstimator,
    run_simulation,
    run_serving,
    run_training,
    sweep_from_names,
)
from .utils.units import parse_size
from .workloads import DEFAULT_REGISTRY as WORKLOAD_REGISTRY
from .workloads.inference_request import DEFAULT_REGISTRY as INFERENCE_WORKLOAD_REGISTRY
from .workloads.training_job import DEFAULT_REGISTRY as TRAINING_WORKLOAD_REGISTRY


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hardware", required=True, help="Hardware preset name")
    parser.add_argument("--model", required=True, help="Model preset name")
    parser.add_argument("--workload", default="swe-agent", help="Workload preset name")
    parser.add_argument("--harness", default="default", help="Harness preset name")
    parser.add_argument("--context", default="32768", help="Context length (supports 8K, 1M, etc.)")
    parser.add_argument("--precision", default="FP8", help="Weight precision")
    parser.add_argument("--kv-precision", default="FP8", help="KV precision")
    parser.add_argument("--optimization", default="baseline", choices=list(OPTIMIZATION_PRESETS.keys()))
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", help="Output file path")


def _add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cluster", help="Cluster preset name (enables topology-aware simulation)")


def _build_simulation_config(args) -> SimulationConfig:
    return SimulationConfig(
        hardware=HW_REGISTRY.get(args.hardware),
        model=MODEL_REGISTRY.get(args.model),
        workload=WORKLOAD_REGISTRY.get(args.workload),
        harness=AgentHarnessSpec(name=args.harness),
        target_context_tokens=parse_size(args.context),
        precision=args.precision,
        kv_precision=args.kv_precision,
        tp=args.tp,
        pp=args.pp,
        batch_size=args.batch_size,
        optimization=OPTIMIZATION_PRESETS[args.optimization],
        random_seed=args.seed,
    )


def _cluster_from_args(args) -> Optional[ClusterSpec]:
    if args.cluster:
        return DEFAULT_CLUSTER_REGISTRY.get(args.cluster)
    return None


def _kv_offload_from_args(args) -> KVOffloadConfig:
    """Build KVOffloadConfig from CLI args or preset name."""
    from .hardware import offload_tiers

    preset_name = getattr(args, "kv_offload_tiers", None)
    if not preset_name:
        return KVOffloadConfig()
    preset_attr = preset_name.replace("-", "_").upper()
    tiers = getattr(offload_tiers, preset_attr, None)
    if tiers is None:
        raise ValueError(
            f"Unknown KV offload preset: {preset_name}. "
            f"Available: {', '.join(sorted([n.lower().replace('_', '-') for n in dir(offload_tiers) if n.endswith('_LIKE') or n.startswith('HBM_') or n == 'NO_OFFLOAD']))}"
        )
    return KVOffloadConfig(tiers=list(tiers))


def cmd_run(args) -> int:
    config = _build_simulation_config(args)
    cluster = _cluster_from_args(args)
    # run_simulation is single-node; cluster not yet wired here
    result = run_simulation(config)
    output = to_json(result)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Saved result to {args.output}")
    else:
        print(output)
    return 0


def cmd_sweep(args) -> int:
    results = sweep_from_names(
        hardware_names=args.hardware.split(","),
        model_names=args.model.split(","),
        workload_name=args.workload,
        context_tokens=[parse_size(c) for c in args.context.split(",")],
        optimization_names=args.optimization.split(","),
        harness_name=args.harness,
        precision=args.precision,
        kv_precision=args.kv_precision,
        seed=args.seed,
    )
    data = [r.to_dict() for r in results]
    output = json.dumps(data, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Saved {len(data)} sweep results to {args.output}")
    else:
        print(output)
    return 0


def cmd_capacity(args) -> int:
    estimator = CapacityEstimator(
        model=MODEL_REGISTRY.get(args.model),
        hardware=HW_REGISTRY.get(args.hardware),
        precision=args.precision,
        kv_precision=args.kv_precision,
        tp=args.tp,
        pp=args.pp,
        batch_size=args.batch_size,
    )
    result = estimator.estimate(parse_size(args.context))
    output = to_json(result)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Saved capacity result to {args.output}")
    else:
        print(output)
    return 0


def cmd_cluster_capacity(args) -> int:
    cluster = DEFAULT_CLUSTER_REGISTRY.get(args.cluster)
    estimator = ClusterCapacityEstimator(
        model=MODEL_REGISTRY.get(args.model),
        hardware=HW_REGISTRY.get(args.hardware),
        cluster=cluster,
        precision=args.precision,
        kv_precision=args.kv_precision,
        optimization=OPTIMIZATION_PRESETS[args.optimization],
    )
    result = estimator.to_simulation_result(parse_size(args.context), batch_size=args.batch_size)
    output = to_json(result)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Saved cluster capacity result to {args.output}")
    else:
        print(output)
    return 0


def cmd_train(args) -> int:
    algorithm = ALGORITHM_REGISTRY.get(args.algorithm)
    model = MODEL_REGISTRY.get(args.model)
    # Override algorithm family if requested
    if model.algorithm_family.name != algorithm.name:
        # Note: ModelSpec is frozen; for CLI convenience we re-bind via object.__setattr__
        # This is safe because algorithm_family is semantically a tag.
        object.__setattr__(model, "algorithm_family", algorithm)

    base_cfg = TRAINING_WORKLOAD_REGISTRY.get(args.workload).training_config
    training_config = TrainingConfig(
        strategy=args.strategy or base_cfg.strategy,
        dataset_tokens=parse_size(args.dataset_tokens),
        epochs=args.epochs,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.sequence_length,
        optimizer=args.optimizer,
        gradient_checkpointing=args.gradient_checkpointing,
        zero_stage=args.zero_stage,
        mfu_target=args.mfu_target,
        parallelism=ParallelismConfig(
            dp=args.dp,
            tp=args.tp,
            pp=args.pp,
            ep=args.ep,
            sp=args.sp,
        ),
    )
    result = run_training(
        model=model,
        hardware=HW_REGISTRY.get(args.hardware),
        training_config=training_config,
        precision=args.precision,
        cluster=_cluster_from_args(args),
    )
    output = to_json(result)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Saved training result to {args.output}")
    else:
        print(output)
    return 0


def cmd_serve(args) -> int:
    algorithm = ALGORITHM_REGISTRY.get(args.algorithm)
    model = MODEL_REGISTRY.get(args.model)
    if model.algorithm_family.name != algorithm.name:
        object.__setattr__(model, "algorithm_family", algorithm)

    base_cfg = INFERENCE_WORKLOAD_REGISTRY.get(args.workload).service_config

    # Build PD config
    pd_config = PDConfig(
        enabled=args.pd_enabled,
        prefill_gpu_count=args.prefill_gpus,
        decode_gpu_count=args.decode_gpus,
        kv_transfer_bw_gb_s=args.kv_transfer_bw,
        kv_transfer_latency_us=args.kv_transfer_latency_us,
        transfer_chunk_size_mb=args.transfer_chunk_size_mb,
        async_prefetch=args.async_prefetch,
    )

    # Build AFD config
    afd_config = AFDConfig(
        enabled=args.afd_enabled,
        attention_gpu_count=args.attention_gpus,
        ffn_gpu_count=args.ffn_gpus,
        decode_gpu_count=args.afd_decode_gpus,
        activation_transfer_bw_gb_s=args.activation_transfer_bw,
        activation_transfer_latency_us=args.activation_transfer_latency_us,
    )

    # Build optimization with optional KV offload tiers
    optimization = OPTIMIZATION_PRESETS[args.optimization]
    kv_offload = _kv_offload_from_args(args)
    if kv_offload.tiers:
        optimization = OptimizationConfig(
            **{**optimization.__dict__, "kv_offload": kv_offload}
        )

    service_config = InferenceServiceConfig(
        arrival_rate_per_sec=args.arrival_rate,
        arrival_distribution=args.arrival_distribution,
        target_ttft_ms=args.target_ttft,
        target_tpot_ms=args.target_tpot,
        max_batch_size=args.max_batch_size,
        max_queue_len=args.max_queue_len,
        prefill_decode_disaggregation=args.pd_enabled or args.prefill_decode_disaggregation,
        pd_config=pd_config,
        afd_config=afd_config,
        request_length_mean=args.request_length_mean,
        request_length_std=args.request_length_std,
        output_length_mean=args.output_length_mean,
        output_length_std=args.output_length_std,
        simulation_duration_seconds=args.simulation_duration,
    )
    result = run_serving(
        model=model,
        hardware=HW_REGISTRY.get(args.hardware),
        service_config=service_config,
        precision=args.precision,
        kv_precision=args.kv_precision,
        gpu_count=args.gpu_count,
        cluster=_cluster_from_args(args),
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        optimization=optimization,
        seed=args.seed,
    )
    output = to_json(result)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Saved serving result to {args.output}")
    else:
        print(output)
    return 0


def cmd_compare_hardware(args) -> int:
    results = sweep_from_names(
        hardware_names=args.hardware.split(","),
        model_names=[args.model],
        workload_name=args.workload,
        context_tokens=[parse_size(args.context)],
        optimization_names=[args.optimization],
        harness_name=args.harness,
        precision=args.precision,
        kv_precision=args.kv_precision,
        seed=args.seed,
    )
    md = to_markdown(results, title=f"Hardware Comparison: {args.model} @ {args.context}")
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Saved comparison report to {args.output}")
    else:
        print(md)
    return 0


def cmd_analyze_trace(args) -> int:
    from .profiling.bridge import TraceAnalyzer

    analyzer = TraceAnalyzer(args.trace)
    result = analyzer.analyze()
    if args.output:
        Path(args.output).write_text(to_markdown([result]), encoding="utf-8")
        print(f"Saved trace report to {args.output}")
    else:
        print(to_markdown([result]))
    return 0


def cmd_report(args) -> int:
    raw = Path(args.input).read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, list):
        md = to_markdown([SimulationResult.from_dict(d) for d in data])
    else:
        md = to_markdown([SimulationResult.from_dict(data)])
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Saved report to {args.output}")
    else:
        print(md)
    return 0


def cmd_list_hardware(args) -> int:
    specs = HW_REGISTRY.list(
        vendor=args.vendor,
        future_only=args.future_only,
        released_only=args.released_only,
    )
    print(f"{'Name':<16} {'Vendor':<8} {'Kind':<6} {'Mem GB':<10} {'FP16 TF':<10} {'FP8 TF':<10} {'Year':<6} Future")
    print("-" * 80)
    for s in specs:
        print(
            f"{s.name:<16} {s.vendor:<8} {s.kind:<6} {s.memory_gb:<10.0f} "
            f"{s.fp16_tflops:<10.0f} {s.fp8_tflops:<10.0f} {s.release_year:<6} {s.is_future}"
        )
    return 0


def cmd_list_topologies(args) -> int:
    specs = DEFAULT_TOPOLOGY_REGISTRY.list()
    print(f"{'Name':<20} {'Type':<16} {'GPUs/Node':<12} {'Intra GB/s':<12} {'Inter GB/s':<12} {'NICs':<6} {'Oversub':<10}")
    print("-" * 100)
    for t in specs:
        print(
            f"{t.name:<20} {t.topology_type:<16} {t.gpus_per_node:<12} "
            f"{t.intra_node_bw_gb_s:<12.0f} {t.inter_node_bw_gb_s:<12.0f} "
            f"{t.nics_per_node:<6} {t.oversubscription_ratio:<10.1f}"
        )
    return 0


def cmd_list_offload_tiers(args) -> int:
    from .hardware import offload_tiers

    presets = [
        ("none", []),
        ("hbm-only", offload_tiers.HBM_ONLY),
        ("hbm-dram", offload_tiers.HBM_DRAM),
        ("hbm-dram-ssd", offload_tiers.HBM_DRAM_SSD),
        ("hbm-icms", offload_tiers.HBM_ICMS),
        ("hbm-cxl", offload_tiers.HBM_CXL),
        ("mooncake-like", offload_tiers.MOONCAKE_LIKE),
        ("lmcache-like", offload_tiers.LMCACHE_LIKE),
    ]
    for name, tiers in presets:
        print(f"{name}")
        for t in tiers:
            print(
                f"  {t.name:<16} cap={t.capacity_gb:>8.0f}GB "
                f"bw={t.bandwidth_gb_s:>6.1f}GB/s lat={t.latency_us:>6.1f}us"
            )
    return 0


def cmd_list_clusters(args) -> int:
    specs = DEFAULT_CLUSTER_REGISTRY.list()
    print(f"{'Name':<24} {'Topology':<20} {'Nodes':<8} {'GPUs/Node':<12} {'Total GPUs':<12}")
    print("-" * 80)
    for c in specs:
        print(
            f"{c.name:<24} {c.topology.name:<20} {c.node_count:<8} "
            f"{c.gpus_per_node:<12} {c.total_gpus:<12}"
        )
    return 0


def cmd_list_models(args) -> int:
    specs = MODEL_REGISTRY.list(architecture=args.architecture)
    print(f"{'Name':<16} {'Arch':<8} {'Total B':<10} {'Active B':<10} {'Layers':<8} {'d_model':<10}")
    print("-" * 70)
    for s in specs:
        print(
            f"{s.name:<16} {s.architecture:<8} {s.total_params_b:<10.0f} "
            f"{s.active_params_b:<10.0f} {s.n_layers:<8} {s.d_model:<10}"
        )
    return 0


def cmd_list_workloads(args) -> int:
    specs = WORKLOAD_REGISTRY.list()
    for s in specs:
        print(f"{s.name:<16} steps={s.avg_steps:.0f}±{s.step_std:.0f} ctx={s.context_limit} {s.description}")
    return 0


def cmd_list_algorithms(args) -> int:
    specs = ALGORITHM_REGISTRY.list()
    print(f"{'Name':<18} {'Attention':<12} {'KV Cache':<10} {'KV Scaling':<12} Notes")
    print("-" * 90)
    for s in specs:
        print(
            f"{s.name:<18} {s.attention_complexity:<12} {str(s.has_kv_cache):<10} "
            f"{s.kv_scaling:<12} {s.notes}"
        )
    return 0


def cmd_list_benchmarks(args) -> int:
    fixtures = BENCHMARK_REGISTRY.list(domain=args.domain)
    print(f"{'Name':<30} {'Domain':<10} {'Model':<12} {'Hardware':<20} Source")
    print("-" * 110)
    for f in fixtures:
        hw = ", ".join(f.hardware_names)
        print(f"{f.name:<30} {f.domain:<10} {f.model_name:<12} {hw:<20} {f.source}")
    return 0


def cmd_benchmark(args) -> int:
    fixture = BENCHMARK_REGISTRY.get(args.name)
    engine = CalibrationEngine(CalibrationConfig(domain=fixture.domain))
    result = engine.evaluate_fixture(fixture)
    output = {
        "name": result["name"],
        "domain": result["domain"],
        "observed": result["observed"],
        "predicted": result["predicted"],
        "errors": result["errors"],
    }
    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Saved benchmark result to {args.output}")
    else:
        print(text)
    return 0


def cmd_calibrate(args) -> int:
    config = CalibrationConfig(
        domain=args.domain,
        fit_params=args.fit_params.split(",") if args.fit_params else [],
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )
    engine = CalibrationEngine(config)
    report = engine.fit(BENCHMARK_REGISTRY)

    if args.output:
        Path(args.output).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved calibration report to {args.output}")
    else:
        print(format_report(report))
    return 0


def cmd_profile(args) -> int:
    orchestrator = ProfilingOrchestrator()
    if args.trace:
        report = orchestrator.profile_trace(
            args.trace,
            hardware_name=args.hardware,
            model_name=args.model,
        )
    elif args.input:
        raw = Path(args.input).read_text(encoding="utf-8")
        result = SimulationResult.from_dict(json.loads(raw))
        report = orchestrator.profile_simulation(result)
    else:
        print("Error: --trace or --input required", file=sys.stderr)
        return 1

    md = _profiling_report_md(report)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Saved profile report to {args.output}")
    else:
        print(md)
    return 0


def _profiling_report_md(report) -> str:
    lines = ["# Profiling Report", ""]
    for layer_name, data in report.layers.items():
        lines.append(f"## {layer_name.capitalize()} Layer")
        lines.append("")
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"- **{key}**: {value}")
            elif isinstance(value, float):
                lines.append(f"- **{key}**: {value:.4f}")
            else:
                lines.append(f"- **{key}**: {value}")
        lines.append("")

    if report.correlations:
        lines.append("## Cross-Layer Correlations")
        lines.append("")
        for key, value in report.correlations.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    if report.recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-sim",
        description="Full-stack simulation platform for AI agent workloads and AI infrastructure.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run a single simulation")
    _add_common_args(p_run)
    _add_cluster_args(p_run)

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Grid sweep over parameters")
    p_sweep.add_argument("--hardware", required=True, help="Comma-separated hardware names")
    p_sweep.add_argument("--model", required=True, help="Comma-separated model names")
    p_sweep.add_argument("--workload", default="swe-agent", help="Workload preset name")
    p_sweep.add_argument("--harness", default="default", help="Harness preset name")
    p_sweep.add_argument("--context", required=True, help="Comma-separated context lengths")
    p_sweep.add_argument("--precision", default="FP8", help="Weight precision")
    p_sweep.add_argument("--kv-precision", default="FP8", help="KV precision")
    p_sweep.add_argument("--optimization", required=True, help="Comma-separated optimization names")
    p_sweep.add_argument("--tp", type=int, default=8)
    p_sweep.add_argument("--pp", type=int, default=1)
    p_sweep.add_argument("--batch-size", type=int, default=1)
    p_sweep.add_argument("--seed", type=int, default=42)
    p_sweep.add_argument("--output", help="Output JSON file")

    # capacity
    p_cap = subparsers.add_parser("capacity", help="Capacity estimation")
    _add_common_args(p_cap)
    _add_cluster_args(p_cap)

    # cluster-capacity
    p_cc = subparsers.add_parser("cluster-capacity", help="Cluster-level distributed capacity estimation")
    p_cc.add_argument("--hardware", required=True, help="Hardware preset name")
    p_cc.add_argument("--model", required=True, help="Model preset name")
    p_cc.add_argument("--cluster", required=True, help="Cluster preset name")
    p_cc.add_argument("--context", default="32768", help="Context length")
    p_cc.add_argument("--precision", default="FP8", help="Weight precision")
    p_cc.add_argument("--kv-precision", default="FP8", help="KV precision")
    p_cc.add_argument("--optimization", default="baseline", choices=list(OPTIMIZATION_PRESETS.keys()))
    p_cc.add_argument("--batch-size", type=int, default=1, help="Batch size")
    p_cc.add_argument("--output", help="Output JSON file")

    # train
    p_train = subparsers.add_parser("train", help="Training simulation")
    p_train.add_argument("--hardware", required=True, help="Hardware preset name")
    p_train.add_argument("--model", required=True, help="Model preset name")
    p_train.add_argument("--algorithm", required=True, help="Algorithm family name")
    p_train.add_argument("--workload", default="pretrain", help="Training workload preset")
    p_train.add_argument("--strategy", help="Training strategy override")
    p_train.add_argument("--dataset-tokens", required=True, help="Dataset size (supports 1T, 10B)")
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--global-batch-size", type=int, default=4096)
    p_train.add_argument("--micro-batch-size", type=int, default=1)
    p_train.add_argument("--sequence-length", type=int, default=4096)
    p_train.add_argument("--optimizer", default="adamw")
    p_train.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p_train.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p_train.add_argument("--zero-stage", type=int, default=1)
    p_train.add_argument("--mfu-target", type=float, default=0.35)
    p_train.add_argument("--dp", type=int, default=1)
    p_train.add_argument("--tp", type=int, default=8)
    p_train.add_argument("--pp", type=int, default=1)
    p_train.add_argument("--ep", type=int, default=1)
    p_train.add_argument("--sp", type=int, default=1)
    p_train.add_argument("--precision", default="FP8")
    p_train.add_argument("--cluster", help="Cluster preset name (topology-aware communication)")
    p_train.add_argument("--output", help="Output JSON file")

    # serve
    p_serve = subparsers.add_parser("serve", help="Inference serving simulation")
    p_serve.add_argument("--hardware", required=True, help="Hardware preset name")
    p_serve.add_argument("--model", required=True, help="Model preset name")
    p_serve.add_argument("--algorithm", required=True, help="Algorithm family name")
    p_serve.add_argument("--workload", default="chat", help="Inference workload preset")
    p_serve.add_argument("--arrival-rate", type=float, default=10.0)
    p_serve.add_argument("--arrival-distribution", default="poisson")
    p_serve.add_argument("--target-ttft", type=float, default=2000.0)
    p_serve.add_argument("--target-tpot", type=float, default=50.0)
    p_serve.add_argument("--max-batch-size", type=int, default=64)
    p_serve.add_argument("--max-queue-len", type=int, default=32)
    p_serve.add_argument("--prefill-decode-disaggregation", action="store_true", help="Deprecated: use --pd-enabled")
    p_serve.add_argument("--pd-enabled", action="store_true", help="Enable Prefill-Decode disaggregation")
    p_serve.add_argument("--prefill-gpus", type=int, default=0, help="GPUs for prefill pool (0 = auto)")
    p_serve.add_argument("--decode-gpus", type=int, default=0, help="GPUs for decode pool (0 = auto)")
    p_serve.add_argument("--kv-transfer-bw", type=float, default=200.0, help="KV transfer bandwidth GB/s")
    p_serve.add_argument("--kv-transfer-latency-us", type=float, default=10.0, help="KV transfer latency us")
    p_serve.add_argument("--transfer-chunk-size-mb", type=float, default=64.0, help="KV transfer chunk size MB")
    p_serve.add_argument("--async-prefetch", action="store_true", help="Async KV prefetch")
    p_serve.add_argument("--afd-enabled", action="store_true", help="Enable Attention-FFN-Decode disaggregation")
    p_serve.add_argument("--attention-gpus", type=int, default=0, help="GPUs for attention pool (AFD)")
    p_serve.add_argument("--ffn-gpus", type=int, default=0, help="GPUs for FFN/expert pool (AFD)")
    p_serve.add_argument("--afd-decode-gpus", type=int, default=0, help="GPUs for decode pool (AFD)")
    p_serve.add_argument("--activation-transfer-bw", type=float, default=200.0, help="AFD activation transfer bandwidth GB/s")
    p_serve.add_argument("--activation-transfer-latency-us", type=float, default=5.0, help="AFD activation transfer latency us")
    p_serve.add_argument("--kv-offload-tiers", help="KV offload preset: none, hbm-dram, hbm-dram-ssd, hbm-icms, hbm-cxl, mooncake-like, lmcache-like")
    p_serve.add_argument("--request-length-mean", type=int, default=4096)
    p_serve.add_argument("--request-length-std", type=int, default=2048)
    p_serve.add_argument("--output-length-mean", type=int, default=512)
    p_serve.add_argument("--output-length-std", type=int, default=256)
    p_serve.add_argument("--simulation-duration", type=float, default=60.0)
    p_serve.add_argument("--gpu-count", type=int, default=1)
    p_serve.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    p_serve.add_argument("--pp", type=int, default=1, help="Pipeline parallelism")
    p_serve.add_argument("--cp", type=int, default=1, help="Context/sequence parallelism")
    p_serve.add_argument("--cluster", help="Cluster preset name (enables topology-aware distributed serving)")
    p_serve.add_argument("--optimization", default="baseline", choices=list(OPTIMIZATION_PRESETS.keys()), help="Optimization preset")
    p_serve.add_argument("--precision", default="FP8")
    p_serve.add_argument("--kv-precision", default="FP8")
    p_serve.add_argument("--seed", type=int, default=42)
    p_serve.add_argument("--output", help="Output JSON file")

    # compare-hardware
    p_cmp = subparsers.add_parser("compare-hardware", help="Compare multiple hardware configs")
    p_cmp.add_argument("--hardware", required=True, help="Comma-separated hardware names")
    p_cmp.add_argument("--model", required=True, help="Model name")
    p_cmp.add_argument("--workload", default="swe-agent")
    p_cmp.add_argument("--harness", default="default")
    p_cmp.add_argument("--context", default="32768")
    p_cmp.add_argument("--precision", default="FP8")
    p_cmp.add_argument("--kv-precision", default="FP8")
    p_cmp.add_argument("--optimization", default="baseline", choices=list(OPTIMIZATION_PRESETS.keys()))
    p_cmp.add_argument("--tp", type=int, default=8)
    p_cmp.add_argument("--pp", type=int, default=1)
    p_cmp.add_argument("--batch-size", type=int, default=1)
    p_cmp.add_argument("--cluster", help="Cluster preset name")
    p_cmp.add_argument("--seed", type=int, default=42)
    p_cmp.add_argument("--output", help="Output markdown file")

    # analyze-trace
    p_trace = subparsers.add_parser("analyze-trace", help="Analyze a legacy trace.jsonl")
    p_trace.add_argument("--trace", required=True, help="Path to trace.jsonl")
    p_trace.add_argument("--output", help="Output markdown file")

    # report
    p_report = subparsers.add_parser("report", help="Generate Markdown report from JSON result")
    p_report.add_argument("--input", required=True, help="Input JSON file")
    p_report.add_argument("--output", help="Output markdown file")

    # list-hardware
    p_lhw = subparsers.add_parser("list-hardware", help="List hardware presets")
    p_lhw.add_argument("--vendor", help="Filter by vendor")
    p_lhw.add_argument("--future-only", action="store_true")
    p_lhw.add_argument("--released-only", action="store_true")

    # list-topologies
    subparsers.add_parser("list-topologies", help="List network topology presets")

    # list-offload-tiers
    subparsers.add_parser("list-offload-tiers", help="List KV offload tier presets")

    # list-clusters
    subparsers.add_parser("list-clusters", help="List cluster presets")

    # list-models
    p_lm = subparsers.add_parser("list-models", help="List model presets")
    p_lm.add_argument("--architecture", help="Filter by architecture (dense/moe)")

    # list-workloads
    subparsers.add_parser("list-workloads", help="List workload presets")

    # list-algorithms
    subparsers.add_parser("list-algorithms", help="List algorithm family presets")

    # list-benchmarks
    p_lb = subparsers.add_parser("list-benchmarks", help="List benchmark fixtures")
    p_lb.add_argument("--domain", choices=["training", "serving", "capacity"], help="Filter by domain")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run a single benchmark fixture")
    p_bench.add_argument("--name", required=True, help="Benchmark fixture name")
    p_bench.add_argument("--output", help="Output JSON file")

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Calibrate simulation constants against benchmarks")
    p_cal.add_argument("--domain", default="all", choices=["training", "serving", "capacity", "all"], help="Domain to calibrate")
    p_cal.add_argument("--fit-params", help="Comma-separated parameter names to fit")
    p_cal.add_argument("--max-iterations", type=int, default=20, help="Max coordinate-descent iterations")
    p_cal.add_argument("--tolerance", type=float, default=0.01, help="Improvement tolerance")
    p_cal.add_argument("--output", help="Output JSON or Markdown file")

    # profile
    p_prof = subparsers.add_parser("profile", help="Multi-layer profiling of a trace or simulation result")
    p_prof.add_argument("--trace", help="Path to trace.jsonl")
    p_prof.add_argument("--input", help="Path to SimulationResult JSON")
    p_prof.add_argument("--hardware", help="Hardware preset name (for trace override)")
    p_prof.add_argument("--model", help="Model preset name (for trace override)")
    p_prof.add_argument("--output", help="Output Markdown file")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "run": cmd_run,
        "sweep": cmd_sweep,
        "capacity": cmd_capacity,
        "cluster-capacity": cmd_cluster_capacity,
        "train": cmd_train,
        "serve": cmd_serve,
        "compare-hardware": cmd_compare_hardware,
        "analyze-trace": cmd_analyze_trace,
        "report": cmd_report,
        "list-hardware": cmd_list_hardware,
        "list-topologies": cmd_list_topologies,
        "list-offload-tiers": cmd_list_offload_tiers,
        "list-clusters": cmd_list_clusters,
        "list-models": cmd_list_models,
        "list-workloads": cmd_list_workloads,
        "list-algorithms": cmd_list_algorithms,
        "list-benchmarks": cmd_list_benchmarks,
        "benchmark": cmd_benchmark,
        "calibrate": cmd_calibrate,
        "profile": cmd_profile,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
