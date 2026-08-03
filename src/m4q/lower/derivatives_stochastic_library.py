from __future__ import annotations

import math

from scipy.stats import norm


def measure_change_density_gap(*, theta: float, observation: float) -> float:
    """Compare the exponential density with SciPy's Gaussian density ratio."""
    transparent = math.exp(theta * observation - 0.5 * theta * theta)
    library = math.exp(
        float(norm.logpdf(observation - theta) - norm.logpdf(observation))
    )
    return abs(transparent - library)


__all__ = ["measure_change_density_gap"]
