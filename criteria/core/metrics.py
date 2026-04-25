from __future__ import annotations

from math import hypot
from statistics import median


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return hypot(ax - bx, ay - by)


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def med(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = avg(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5

