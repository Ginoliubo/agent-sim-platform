"""Tests for CLI."""

import json
import tempfile
from pathlib import Path

import pytest

from agent_sim_platform.cli import main


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_list_topologies(capsys):
    assert main(["list-topologies"]) == 0
    captured = capsys.readouterr()
    assert "nvlink-domain-8" in captured.out
    assert "rail-optimized" in captured.out


def test_cli_list_clusters(capsys):
    assert main(["list-clusters"]) == 0
    captured = capsys.readouterr()
    assert "fat-tree-256-h100" in captured.out


def test_cli_cluster_capacity(capsys):
    assert main([
        "cluster-capacity",
        "--hardware", "B200",
        "--model", "1T-Dense",
        "--cluster", "fat-tree-1024-h100",
        "--context", "10M",
        "--optimization", "layer3",
    ]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["feasible"] is True
    assert data["gpu_count"] > 1
    assert "cp_comm_time_ms" in data["metadata"]


def test_cli_serve_distributed(capsys):
    assert main([
        "serve",
        "--hardware", "B200",
        "--model", "1T-Dense",
        "--algorithm", "dense",
        "--cluster", "fat-tree-1024-h100",
        "--tp", "8",
        "--pp", "4",
        "--cp", "64",
        "--request-length-mean", "10000000",
        "--output-length-mean", "1000",
        "--arrival-rate", "1",
        "--max-batch-size", "1",
        "--simulation-duration", "1",
        "--optimization", "layer3",
    ]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "cp_comm_time_ms" in data["metadata"]
    assert data["metadata"]["tp"] == 8
    assert data["metadata"]["pp"] == 4
    assert data["metadata"]["cp"] == 64


def test_cli_list_offload_tiers(capsys):
    assert main(["list-offload-tiers"]) == 0
    captured = capsys.readouterr()
    assert "gpu_hbm" in captured.out
    assert "icms" in captured.out


def test_cli_serve_pd_kv_offload(capsys):
    assert main([
        "serve",
        "--hardware", "B200",
        "--model", "1T-MoE-MLA",
        "--algorithm", "mla",
        "--cluster", "fat-tree-1024-h100",
        "--tp", "8",
        "--pp", "4",
        "--cp", "32",
        "--pd-enabled",
        "--prefill-gpus", "512",
        "--decode-gpus", "512",
        "--kv-transfer-bw", "200",
        "--kv-offload-tiers", "mooncake-like",
        "--optimization", "layer3",
        "--request-length-mean", "10000000",
        "--output-length-mean", "1000",
        "--arrival-rate", "1",
        "--max-batch-size", "1",
        "--simulation-duration", "1",
    ]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["feasible"] is True
    assert data["metadata"]["pd_enabled"] is True
    assert data["metadata"]["kv_transfer_time_per_request_ms"] > 0


def test_cli_list_hardware(capsys):
    assert main(["list-hardware"]) == 0
    captured = capsys.readouterr()
    assert "H100-SXM5" in captured.out


def test_cli_list_models(capsys):
    assert main(["list-models"]) == 0
    captured = capsys.readouterr()
    assert "10T-MoE" in captured.out


def test_cli_run(capsys):
    assert main(["run", "--hardware", "H100-SXM5", "--model", "1T-MoE", "--context", "32K"]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["feasible"] in (True, False)
    assert data["gpu_count"] >= 1


def test_cli_run_output_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "result.json"
        assert main([
            "run",
            "--hardware", "H100-SXM5",
            "--model", "1T-MoE",
            "--context", "32K",
            "--output", str(path),
        ]) == 0
        assert path.exists()
        data = json.loads(path.read_text())
        assert "bottleneck" in data


def test_cli_capacity(capsys):
    assert main([
        "capacity", "--hardware", "H100-SXM5", "--model", "70B-Dense", "--context", "32K"
    ]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["feasible"] is True


def test_cli_list_algorithms(capsys):
    assert main(["list-algorithms"]) == 0
    captured = capsys.readouterr()
    assert "dense" in captured.out
    assert "mamba" in captured.out


def test_cli_train(capsys):
    assert main([
        "train",
        "--hardware", "H100-SXM5",
        "--model", "70B-Dense",
        "--algorithm", "dense",
        "--dataset-tokens", "1B",
        "--dp", "8",
        "--tp", "8",
    ]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["feasible"] in (True, False)
    assert "step_time_seconds" in data["metadata"]


def test_cli_train_output_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "training.json"
        assert main([
            "train",
            "--hardware", "H100-SXM5",
            "--model", "70B-Dense",
            "--algorithm", "dense",
            "--dataset-tokens", "1B",
            "--dp", "8",
            "--tp", "8",
            "--output", str(path),
        ]) == 0
        assert path.exists()
        data = json.loads(path.read_text())
        assert "mfu" in data["metadata"]


def test_cli_serve(capsys):
    assert main([
        "serve",
        "--hardware", "H100-SXM5",
        "--model", "70B-Dense",
        "--algorithm", "dense",
        "--arrival-rate", "2",
        "--max-batch-size", "4",
        "--simulation-duration", "10",
        "--gpu-count", "8",
    ]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "requests_completed" in data["metadata"]
    assert "ttft_p99_ms" in data["metadata"]


def test_cli_serve_output_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "serving.json"
        assert main([
            "serve",
            "--hardware", "H100-SXM5",
            "--model", "70B-Dense",
            "--algorithm", "dense",
            "--arrival-rate", "2",
            "--max-batch-size", "4",
            "--simulation-duration", "10",
            "--gpu-count", "8",
            "--output", str(path),
        ]) == 0
        assert path.exists()
        data = json.loads(path.read_text())
        assert "tpot_p99_ms" in data["metadata"]


def test_cli_compare_hardware(capsys):
    assert main([
        "compare-hardware",
        "--hardware", "H100-SXM5,B200",
        "--model", "1T-MoE",
        "--context", "32K",
    ]) == 0
    captured = capsys.readouterr()
    assert "Comparison Table" in captured.out
    assert "H100-SXM5" in captured.out
    assert "B200" in captured.out


def test_cli_list_benchmarks(capsys):
    assert main(["list-benchmarks"]) == 0
    captured = capsys.readouterr()
    assert "llama2_70b_pretrain" in captured.out
    assert "vllm_llama70b_serving" in captured.out


def test_cli_list_benchmarks_filter(capsys):
    assert main(["list-benchmarks", "--domain", "training"]) == 0
    captured = capsys.readouterr()
    assert "llama2_70b_pretrain" in captured.out
    assert "vllm_llama70b_serving" not in captured.out


def test_cli_benchmark(capsys):
    assert main([
        "benchmark",
        "--name", "capacity_llama70b_4k",
    ]) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["name"] == "capacity_llama70b_4k"
    assert "predicted" in data
    assert "errors" in data


def test_cli_calibrate(capsys):
    assert main([
        "calibrate",
        "--domain", "capacity",
        "--fit-params", "default_decode_utilization",
        "--max-iterations", "2",
    ]) == 0
    captured = capsys.readouterr()
    assert "Calibration Report" in captured.out


def test_cli_profile_trace(capsys):
    trace_path = Path(__file__).parent / "fixtures" / "trace.jsonl"
    assert main([
        "profile",
        "--trace", str(trace_path),
        "--hardware", "H100-SXM5",
        "--model", "70B-Dense",
    ]) == 0
    captured = capsys.readouterr()
    assert "Profiling Report" in captured.out
    assert "Software Layer" in captured.out


def test_cli_profile_input(tmp_path):
    result = {
        "config": {
            "hardware": {
                "name": "H100-SXM5",
                "vendor": "nvidia",
                "kind": "gpu",
                "memory_gb": 80.0,
                "memory_bw_tb_s": 3.35,
                "fp16_tflops": 989.0,
                "fp8_tflops": 1979.0,
            },
            "model": {
                "name": "70B-Dense",
                "total_params_b": 70,
                "active_params_b": 70,
                "n_layers": 80,
                "d_model": 8192,
                "n_heads": 64,
                "d_head": 128,
            },
            "workload": {"name": "test", "max_steps": 1, "avg_steps": 1.0, "step_std": 0.0, "context_limit": 32768},
            "harness": {"name": "test"},
            "target_context_tokens": 32768,
            "precision": "FP8",
            "optimization": {"name": "baseline"},
        },
        "latency_seconds": 1.0,
        "wall_time_seconds": 1.0,
        "tokens_total": 100,
        "gpu_count": 1,
        "feasible": True,
        "metadata": {},
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert main(["profile", "--input", str(path)]) == 0
