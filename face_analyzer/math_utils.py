from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


EPSILON = 1e-8


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def smoothstep(low: float, high: float, value: float) -> float:
    if high <= low:
        return 1.0 if value >= high else 0.0
    x = clamp01((value - low) / (high - low))
    return x * x * (3.0 - 2.0 * x)


def descending_smoothstep(
    high_quality_edge: float,
    low_quality_edge: float,
    value: float,
) -> float:
    if low_quality_edge <= high_quality_edge:
        return 1.0 if value <= high_quality_edge else 0.0
    x = clamp01(
        (value - high_quality_edge)
        / (low_quality_edge - high_quality_edge)
    )
    return 1.0 - x * x * (3.0 - 2.0 * x)


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-value))


def softmax(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    flattened -= float(np.max(flattened))
    exponentials = np.exp(flattened)
    return (exponentials / max(float(exponentials.sum()), EPSILON)).astype(
        np.float32
    )


def confidence_weighted_average(
    entries: Iterable[tuple[float | None, float, float]],
    *,
    fallback: float,
) -> float:
    total = 0.0
    total_weight = 0.0

    for value, configured_weight, confidence in entries:
        if value is None or not math.isfinite(value):
            continue

        effective_weight = max(0.0, configured_weight) * clamp01(confidence)
        if effective_weight <= 0.0:
            continue

        total += clamp01(value) * effective_weight
        total_weight += effective_weight

    if total_weight <= EPSILON:
        return clamp01(fallback)
    return clamp01(total / total_weight)
