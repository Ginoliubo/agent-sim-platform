"""Hardware specification presets and registry."""

from typing import List, Optional

from .base import HardwareSpec
from .google import GOOGLE_HARDWARE
from .huawei import HUAWEI_HARDWARE
from .nvidia import NVIDIA_HARDWARE

__all__ = [
    "HardwareSpec",
    "HardwareRegistry",
    "ALL_HARDWARE",
    "NVIDIA_HARDWARE",
    "HUAWEI_HARDWARE",
    "GOOGLE_HARDWARE",
]


class HardwareRegistry:
    """Central registry for accelerator presets."""

    def __init__(self, specs=None):
        self._specs: dict = {}
        if specs:
            for spec in specs:
                self.register(spec)

    def register(self, spec: HardwareSpec) -> None:
        """Register a hardware spec."""
        if spec.name in self._specs:
            raise ValueError(f"Hardware '{spec.name}' already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> HardwareSpec:
        """Retrieve a spec by name."""
        if name not in self._specs:
            raise KeyError(f"Hardware '{name}' not found. Available: {list(self._specs.keys())}")
        return self._specs[name]

    def list(
        self,
        vendor: Optional[str] = None,
        kind: Optional[str] = None,
        future_only: bool = False,
        released_only: bool = False,
    ) -> List[HardwareSpec]:
        """List matching hardware specs."""
        results = []
        for spec in self._specs.values():
            if vendor and spec.vendor.lower() != vendor.lower():
                continue
            if kind and spec.kind.lower() != kind.lower():
                continue
            if future_only and not spec.is_future:
                continue
            if released_only and spec.is_future:
                continue
            results.append(spec)
        return sorted(results, key=lambda s: (s.vendor, s.release_year, s.name))

    def names(self) -> List[str]:
        """Return all registered names."""
        return sorted(self._specs.keys())


ALL_HARDWARE = list(NVIDIA_HARDWARE) + list(HUAWEI_HARDWARE) + list(GOOGLE_HARDWARE)
DEFAULT_REGISTRY = HardwareRegistry(ALL_HARDWARE)
