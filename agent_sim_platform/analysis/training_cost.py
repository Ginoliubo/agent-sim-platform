"""Training cost analysis."""

from ..config import DEFAULT_DOLLAR_PER_KWH
from ..data_models import SimulationResult


class TrainingCostAnalyzer:
    """Analyze training cost from a SimulationResult."""

    def __init__(self, result: SimulationResult):
        self.result = result

    def gpu_hours(self) -> float:
        """Total GPU-hours consumed."""
        return self.result.latency_seconds / 3600.0 * self.result.gpu_count

    def rental_cost(self) -> float:
        """Rental cost from hardware cost_per_hour."""
        return self.result.cost_usd

    def energy_cost(self, dollar_per_kwh: float = DEFAULT_DOLLAR_PER_KWH) -> float:
        """Energy cost from TDP and runtime."""
        hours = self.result.latency_seconds / 3600.0
        kwh = self.result.config.hardware.power_w / 1000.0 * self.result.gpu_count * hours
        return kwh * dollar_per_kwh

    def total_cost(self, dollar_per_kwh: float = DEFAULT_DOLLAR_PER_KWH) -> float:
        return self.rental_cost() + self.energy_cost(dollar_per_kwh)

    def cost_per_trillion_tokens(self) -> float:
        """Cost per trillion tokens trained."""
        if self.result.tokens_total <= 0:
            return float("inf")
        return self.result.cost_usd / (self.result.tokens_total / 1e12)

    def report(self) -> dict:
        return {
            "gpu_hours": round(self.gpu_hours(), 2),
            "rental_usd": round(self.rental_cost(), 2),
            "energy_usd": round(self.energy_cost(), 2),
            "total_usd": round(self.total_cost(), 2),
            "per_trillion_tokens_usd": round(self.cost_per_trillion_tokens(), 2),
        }


def analyze(result: SimulationResult) -> dict:
    """Convenience function returning training cost summary."""
    return TrainingCostAnalyzer(result).report()
