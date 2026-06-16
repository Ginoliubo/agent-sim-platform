"""Unified CLI for agent-sim."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .algorithms import DEFAULT_REGISTRY as ALGORITHM_REGISTRY
from .config import OPTIMIZATION_PRESETS
from .data_models import (
    AgentHarnessSpec,
    InferenceServiceConfig,
    ParallelismConfig,
    SimulationConfig,
    SimulationResult,
    TrainingConfig,
)
from .hardware import DEFAULT_REGISTRY as HW_REGISTRY
from .models import DEFAULT_REGISTRY as MODEL_REGISTRY
from .reports.json_reporter import to_json
from .reports.markdown_reporter import to_markdown
from .simulation import (
    CapacityEstimator,
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


def cmd_run(args) -> int:
    config = _build_simulation_config(args)
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
    service_config = InferenceServiceConfig(
        arrival_rate_per_sec=args.arrival_rate,
        arrival_distribution=args.arrival_distribution,
        target_ttft_ms=args.target_ttft,
        target_tpot_ms=args.target_tpot,
        max_batch_size=args.max_batch_size,
        max_queue_len=args.max_queue_len,
        prefill_decode_disaggregation=args.prefill_decode_disaggregation,
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
    p_serve.add_argument("--prefill-decode-disaggregation", action="store_true")
    p_serve.add_argument("--request-length-mean", type=int, default=4096)
    p_serve.add_argument("--request-length-std", type=int, default=2048)
    p_serve.add_argument("--output-length-mean", type=int, default=512)
    p_serve.add_argument("--output-length-std", type=int, default=256)
    p_serve.add_argument("--simulation-duration", type=float, default=60.0)
    p_serve.add_argument("--gpu-count", type=int, default=1)
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

    # list-models
    p_lm = subparsers.add_parser("list-models", help="List model presets")
    p_lm.add_argument("--architecture", help="Filter by architecture (dense/moe)")

    # list-workloads
    subparsers.add_parser("list-workloads", help="List workload presets")

    # list-algorithms
    subparsers.add_parser("list-algorithms", help="List algorithm family presets")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "run": cmd_run,
        "sweep": cmd_sweep,
        "capacity": cmd_capacity,
        "train": cmd_train,
        "serve": cmd_serve,
        "compare-hardware": cmd_compare_hardware,
        "analyze-trace": cmd_analyze_trace,
        "report": cmd_report,
        "list-hardware": cmd_list_hardware,
        "list-models": cmd_list_models,
        "list-workloads": cmd_list_workloads,
        "list-algorithms": cmd_list_algorithms,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
