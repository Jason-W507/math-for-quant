from __future__ import annotations

import numpy as np


def vectorized_static_pnl(
    signs: np.ndarray,
    price_changes: np.ndarray,
    half_spread: float,
    fills: np.ndarray | None = None,
) -> float:
    signs = np.asarray(signs, dtype=float)
    changes = np.asarray(price_changes, dtype=float)
    if signs.shape != changes.shape:
        raise ValueError("simulation arrays must align")
    filled = np.ones(signs.shape, dtype=bool) if fills is None else np.asarray(fills, dtype=bool)
    if filled.shape != signs.shape:
        raise ValueError("fill mask must align with the event stream")
    inventory_changes = np.where(filled, -signs, 0.0)
    mid_before = 100.0 + np.concatenate(([0.0], np.cumsum(changes[:-1])))
    quotes = mid_before + signs * half_spread
    cash = float(np.sum(-inventory_changes * quotes))
    inventory = float(np.sum(inventory_changes))
    terminal_mid = 100.0 + float(np.sum(changes))
    return cash + inventory * terminal_mid


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
