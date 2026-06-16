"""Unit conversion helpers for memory, bandwidth, compute, and time."""

from typing import Union

Number = Union[int, float]


def gb_to_bytes(gb: Number) -> float:
    """Convert GB (decimal, 1 GB = 1e9 bytes) to bytes."""
    return float(gb) * 1e9


def bytes_to_gb(b: Number) -> float:
    """Convert bytes to GB (decimal)."""
    return float(b) / 1e9


def bytes_to_gib(b: Number) -> float:
    """Convert bytes to GiB (binary, 1 GiB = 1024^3 bytes)."""
    return float(b) / (1024**3)


def gib_to_bytes(gib: Number) -> float:
    """Convert GiB to bytes."""
    return float(gib) * (1024**3)


def tb_s_to_bytes_s(tb_s: Number) -> float:
    """Convert TB/s to bytes/s."""
    return float(tb_s) * 1e12


def tflops_to_flops(tflops: Number) -> float:
    """Convert TFLOPS to FLOPS."""
    return float(tflops) * 1e12


def flops_to_tflops(flops: Number) -> float:
    """Convert FLOPS to TFLOPS."""
    return float(flops) / 1e12


def seconds_to_hms(seconds: Number) -> str:
    """Convert seconds to a human-readable 'Hh Mm Ss' string."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def format_bytes(b: Number) -> str:
    """Format bytes to the largest sensible unit."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(b)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} {units[-1]}"


def parse_size(size_str: str) -> int:
    """Parse strings like '8K', '32K', '1M', '10M' into integer token counts.

    Supports K=1e3, M=1e6, G=1e9, B=billion.
    """
    size_str = size_str.strip().upper()
    multipliers = {
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
        "B": 1e9,
    }
    for suffix, mult in multipliers.items():
        if size_str.endswith(suffix):
            return int(float(size_str[:-1]) * mult)
    return int(size_str)
