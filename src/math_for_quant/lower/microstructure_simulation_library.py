from __future__ import annotations

import numpy as np


def vectorized_static_pnl(signs: np.ndarray, price_changes: np.ndarray, half_spread: float) -> float:
    signs = np.asarray(signs, dtype=float)
    changes = np.asarray(price_changes, dtype=float)
    if signs.shape != changes.shape:
        raise ValueError("simulation arrays must align")
    return float(signs.size * half_spread + signs @ changes)


def library_duration_and_beta(
    seconds: np.ndarray, maker_signs: np.ndarray, prices: np.ndarray
) -> tuple[float, float]:
    seconds = np.asarray(seconds, dtype=float)
    signs = np.asarray(maker_signs, dtype=float)[1:]
    changes = np.diff(np.asarray(prices, dtype=float))
    design = np.column_stack([np.ones(signs.size), signs])
    coefficient, *_ = np.linalg.lstsq(design, changes, rcond=None)
    return float(seconds[-1] - seconds[0]), float(coefficient[1])


__all__ = ["library_duration_and_beta", "vectorized_static_pnl"]
