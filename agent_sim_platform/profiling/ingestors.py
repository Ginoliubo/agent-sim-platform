"""Input adapters for profiling: simulation results, traces, summaries."""

import json
from pathlib import Path
from typing import Dict, List

from ..data_models import SimulationResult
from .bridge import TraceAnalyzer


class SimulationIngestor:
    """Ingest a SimulationResult directly."""

    def ingest(self, result: SimulationResult) -> SimulationResult:
        return result


class TraceIngestor:
    """Ingest a legacy trace.jsonl and produce a SimulationResult."""

    def __init__(self, trace_path: str):
        self.trace_path = Path(trace_path)
        self.analyzer = TraceAnalyzer(trace_path)

    def ingest(self) -> SimulationResult:
        return self.analyzer.analyze()

    def raw_records(self) -> List[Dict]:
        return self.analyzer.records


class SummaryIngestor:
    """Ingest a JSON summary file containing observed metrics."""

    def __init__(self, summary_path: str):
        self.summary_path = Path(summary_path)

    def ingest(self) -> Dict:
        return json.loads(self.summary_path.read_text(encoding="utf-8"))


__all__ = [
    "SimulationIngestor",
    "SummaryIngestor",
    "TraceIngestor",
]
