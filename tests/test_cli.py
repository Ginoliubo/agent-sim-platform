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
