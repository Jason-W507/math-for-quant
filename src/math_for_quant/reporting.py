from __future__ import annotations


REPORT_ZERO_TOLERANCE = 1e-10


def stable_gap(value: float | int) -> float:
    """Normalize numerical noise in frozen, cross-platform reports."""

    observed = float(value)
    return 0.0 if abs(observed) < REPORT_ZERO_TOLERANCE else observed
