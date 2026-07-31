from __future__ import annotations

import numpy as np


def library_historical_var_es(losses: np.ndarray, confidence: float) -> tuple[float, float]:
    sample = np.sort(np.asarray(losses, dtype=float))
    value_at_risk = float(np.quantile(sample, confidence, method="lower"))
    tail_count = int(round(sample.size * (1.0 - confidence)))
    if tail_count <= 0 or not np.isclose(tail_count, sample.size * (1.0 - confidence)):
        raise ValueError("library comparison requires an integer tail count")
    return value_at_risk, float(sample[-tail_count:].mean())


__all__ = ["library_historical_var_es"]
