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


def vectorized_feedback_ledger(
    signs: np.ndarray,
    quotes: np.ndarray,
    fill_mask: np.ndarray,
    terminal_midprice: float,
) -> tuple[float, int, int, int]:
    """Revalue an exported quote/fill trace without reimplementing the policy."""
    raw_directions = np.asarray(signs)
    prices = np.asarray(quotes, dtype=float)
    raw_fills = np.asarray(fill_mask)
    if (
        raw_directions.ndim != 1
        or prices.ndim != 1
        or raw_fills.ndim != 1
        or raw_directions.size == 0
        or raw_directions.shape != prices.shape
        or raw_directions.shape != raw_fills.shape
    ):
        raise ValueError("feedback ledger arrays must be nonempty, one-dimensional and aligned")
    if (
        not np.issubdtype(raw_directions.dtype, np.number)
        or np.any(~np.isfinite(raw_directions.astype(float)))
        or np.any(~np.isin(raw_directions, (-1, 1)))
    ):
        raise ValueError("feedback ledger directions must be finite and exactly -1 or 1")
    if raw_fills.dtype != np.bool_:
        raise ValueError("feedback ledger fill mask must contain booleans")
    if (
        np.any(~np.isfinite(prices))
        or np.any(prices <= 0.0)
        or not np.isfinite(terminal_midprice)
        or terminal_midprice <= 0.0
    ):
        raise ValueError("feedback ledger prices must be finite and positive")

    directions = raw_directions.astype(np.int8, copy=False)
    inventory_changes = np.where(raw_fills, -directions, 0)
    inventory_path = np.cumsum(inventory_changes, dtype=np.int64)
    cash = -float(np.sum(inventory_changes * prices))
    position = int(inventory_path[-1])
    maximum = int(np.max(np.abs(inventory_path)))
    fills = int(np.count_nonzero(raw_fills))
    pnl = cash + position * float(terminal_midprice)
    return pnl, fills, position, maximum


__all__ = [
    "library_duration_and_beta",
    "vectorized_feedback_ledger",
    "vectorized_static_pnl",
]
