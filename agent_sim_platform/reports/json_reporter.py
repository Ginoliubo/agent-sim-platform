"""JSON reporter for simulation results."""

import json
from dataclasses import asdict
from typing import Dict, List

from ..data_models import SimulationResult


def to_json(result: SimulationResult, indent: int = 2) -> str:
    """Serialize a single SimulationResult to JSON string."""
    return json.dumps(result.to_dict(), indent=indent, ensure_ascii=False)


def to_json_list(results: List[SimulationResult], indent: int = 2) -> str:
    """Serialize multiple SimulationResults to JSON string."""
    return json.dumps([r.to_dict() for r in results], indent=indent, ensure_ascii=False)


def to_dict(result: SimulationResult) -> Dict:
    """Serialize a single SimulationResult to a plain dictionary."""
    return result.to_dict()
