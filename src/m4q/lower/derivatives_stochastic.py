from __future__ import annotations

import numpy as np

from m4q.lower.derivatives import (
    terminal_singular_theta_energy,
    validate_novikov_exponential_moment,
    validate_risk_neutral_drift,
)


def nested_quadratic_variation(
    fine_increments: np.ndarray,
    *,
    block_sizes: tuple[int, ...],
) -> dict[int, float]:
    """Compute quadratic variation on genuinely nested partitions.

    ``fine_increments`` are Brownian increments on the finest grid.  A block
    size of ``m`` aggregates each consecutive group of ``m`` increments before
    squaring, so every reported partition is built from the same path.
    """
    increments = np.asarray(fine_increments, dtype=float)
    if increments.ndim != 1 or increments.size == 0:
        raise ValueError("quadratic variation requires a nonempty increment vector")
    if not block_sizes or any(size <= 0 for size in block_sizes):
        raise ValueError("nested partition block sizes must be positive")
    result: dict[int, float] = {}
    for size in block_sizes:
        if increments.size % size:
            raise ValueError("nested partition block size must divide the fine grid")
        aggregated = increments.reshape(-1, size).sum(axis=1)
        result[size] = float(aggregated @ aggregated)
    return result


__all__ = [
    "nested_quadratic_variation",
    "terminal_singular_theta_energy",
    "validate_novikov_exponential_moment",
    "validate_risk_neutral_drift",
]
