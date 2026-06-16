"""Calibration report data model and serialization."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CalibrationReport:
    """Result of a calibration run."""

    fixture_results: List[Dict] = field(default_factory=list)
    metric_summary: Dict[str, Dict] = field(default_factory=dict)
    overall_mape: float = 0.0
    overall_rmse: float = 0.0
    fitted_values: Dict[str, float] = field(default_factory=dict)
    iterations: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "fixture_results": self.fixture_results,
            "metric_summary": self.metric_summary,
            "overall_mape": self.overall_mape,
            "overall_rmse": self.overall_rmse,
            "fitted_values": self.fitted_values,
            "iterations": self.iterations,
        }


def format_report(report: CalibrationReport) -> str:
    """Return a concise text summary of a calibration report."""
    lines = [
        "# Calibration Report",
        "",
        f"Overall MAPE: {report.overall_mape:.2%}",
        f"Overall RMSE: {report.overall_rmse:.4f}",
        "",
        "## Metric Summary",
        "",
        "| Metric | MAPE | RMSE | Samples |",
        "|--------|------|------|---------|",
    ]
    for metric, summary in report.metric_summary.items():
        lines.append(
            f"| {metric} | {summary['mape']:.2%} | {summary['rmse']:.4f} | {summary['samples']} |"
        )
    lines.append("")

    if report.fitted_values:
        lines.append("## Fitted Constants")
        lines.append("")
        for name, value in report.fitted_values.items():
            lines.append(f"- **{name}**: {value:.4f}")
        lines.append("")

    lines.append("## Per-Fixture Results")
    lines.append("")
    for result in report.fixture_results:
        lines.append(f"### {result['name']} ({result['domain']})")
        lines.append("")
        for metric, err in result.get("errors", {}).items():
            lines.append(
                f"- {metric}: observed={err['observed']:.4f}, predicted={err['predicted']:.4f}, "
                f"MAPE={err['pct']:.2%}"
            )
        lines.append("")

    return "\n".join(lines)


__all__ = ["CalibrationReport", "format_report"]
