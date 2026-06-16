"""Cost analysis for simulation results."""

from ..config import DEFAULT_DOLLAR_PER_KWH
from ..data_models import SimulationResult


class CostAnalyzer:
    """Estimate energy and rental cost from a SimulationResult."""

    def __init__(self, result: SimulationResult):
        self.result = result

    def rental_cost(self) -> float:
        """Rental cost based on hardware cost_per_hour and GPU count."""
        return self.result.cost_usd

    def energy_cost(self, dollar_per_kwh: float = DEFAULT_DOLLAR_PER_KWH) -> float:
        """Energy cost based on TDP and runtime."""
        hours = self.result.latency_seconds / 3600.0
        kwh = (self.result.config.hardware.power_w / 1000.0) * self.result.gpu_count * hours
        return kwh * dollar_per_kwh

    def total_cost(self, dollar_per_kwh: float = DEFAULT_DOLLAR_PER_KWH) -> float:
        """Rental + energy cost."""
        return self.rental_cost() + self.energy_cost(dollar_per_kwh)

    def cost_per_million_tokens(self) -> float:
        """Cost per million tokens generated."""
        if self.result.tokens_total <= 0:
            return float("inf")
        return self.result.cost_usd / (self.result.tokens_total / 1e6)

    def report(self) -> dict:
        """Return a cost summary dictionary."""
        return {
            "rental_usd": round(self.rental_cost(), 4),
            "energy_usd": round(self.energy_cost(), 4),
            "total_usd": round(self.total_cost(), 4),
            "per_million_tokens_usd": round(self.cost_per_million_tokens(), 4),
        }


def analyze(result: SimulationResult) -> dict:
    """Convenience function returning cost summary."""
    return CostAnalyzer(result).report()
