from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def softmax_with_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be a positive finite number")
    scaled = [float(value) / temperature for value in logits]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def scale_probabilities(
    probabilities: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
    labels = list(probabilities)
    logits = [math.log(max(float(probabilities[label]), 1e-12)) for label in labels]
    scaled = softmax_with_temperature(logits, temperature)
    return dict(zip(labels, scaled, strict=True))


def negative_log_likelihood(
    logits: Sequence[Sequence[float]],
    targets: Sequence[int],
    temperature: float,
) -> float:
    if len(logits) != len(targets) or not logits:
        raise ValueError("logits and targets must have the same non-zero length")
    losses = []
    for row, target in zip(logits, targets, strict=True):
        probabilities = softmax_with_temperature(row, temperature)
        if not 0 <= target < len(probabilities):
            raise ValueError(f"target index is out of range: {target}")
        losses.append(-math.log(max(probabilities[target], 1e-12)))
    return sum(losses) / len(losses)


def fit_temperature(
    logits: Sequence[Sequence[float]],
    targets: Sequence[int],
) -> tuple[float, float, float]:
    """Fit one temperature on held-out logits using golden-section search."""
    if len(logits) != len(targets) or not logits:
        raise ValueError("logits and targets must have the same non-zero length")
    lower = math.log(0.05)
    upper = math.log(10.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)

    for _ in range(80):
        left_loss = negative_log_likelihood(logits, targets, math.exp(left))
        right_loss = negative_log_likelihood(logits, targets, math.exp(right))
        if left_loss <= right_loss:
            upper = right
            right = left
            left = upper - ratio * (upper - lower)
        else:
            lower = left
            left = right
            right = lower + ratio * (upper - lower)

    temperature = math.exp((lower + upper) / 2.0)
    before = negative_log_likelihood(logits, targets, 1.0)
    after = negative_log_likelihood(logits, targets, temperature)
    return temperature, before, after
