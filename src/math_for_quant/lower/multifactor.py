from __future__ import annotations

from datetime import date
import math
from pathlib import Path
import sys

import numpy as np

from math_for_quant.evidence import load_oracle_bundle


def validate_time_boundary(signal_date: str, return_date: str) -> None:
    if date.fromisoformat(signal_date) >= date.fromisoformat(return_date):
        raise ValueError("signal timestamp must precede the realized-return timestamp")


def ols_slope(signal: np.ndarray, future_return: np.ndarray) -> float:
    centered = signal - signal.mean()
    denominator = float(centered @ centered)
    if denominator == 0.0:
        raise ValueError("cross-sectional signal has zero variation")
    return float(centered @ (future_return - future_return.mean()) / denominator)


def ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, so ties are permutation invariant."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    result = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(
        float(left_centered @ left_centered) * float(right_centered @ right_centered)
    )
    if denominator == 0.0:
        raise ValueError("correlation requires nonconstant inputs")
    return float(left_centered @ right_centered / denominator)


def neutralize(
    signal: np.ndarray,
    size: np.ndarray,
    industry: np.ndarray,
    condition_limit: float,
) -> tuple[np.ndarray, float]:
    design = np.column_stack((np.ones(signal.size), size, industry))
    condition = float(np.linalg.cond(design))
    if not math.isfinite(condition) or condition > condition_limit:
        raise ValueError("neutralization design is rank deficient or ill-conditioned")
    residual = signal - design @ np.linalg.lstsq(design, signal, rcond=None)[0]
    exposure = float(np.max(np.abs(design.T @ residual)))
    return residual, exposure


def bh_rejections(p_values: list[float], alpha: float) -> int:
    ordered = sorted(p_values)
    count = len(ordered)
    return max(
        (index for index, value in enumerate(ordered, 1) if value <= alpha * index / count),
        default=0,
    )


def null_search_p_values(seed: int, observations: int, attempts: int) -> list[float]:
    """Run a frozen null search and return Fisher-z two-sided p-values."""
    generator = np.random.default_rng(seed)
    future_return = generator.standard_normal(observations)
    p_values: list[float] = []
    for _ in range(attempts):
        signal = generator.standard_normal(observations)
        coefficient = correlation(signal, future_return)
        bounded = max(-0.999999, min(0.999999, coefficient))
        z_score = math.atanh(bounded) * math.sqrt(observations - 3)
        p_values.append(math.erfc(abs(z_score) / math.sqrt(2.0)))
    return p_values


def main(oracle_path: Path = Path("evidence/lower-ch01/oracle.json")) -> int:
    oracle = load_oracle_bundle(oracle_path)
    fixture = oracle
    tolerance = float(oracle["absolute_tolerance"])

    slopes: list[float] = []
    correlations: list[float] = []
    rank_correlations: list[float] = []
    spread_returns: list[float] = []
    neutral_exposures: list[float] = []
    for period in fixture["periods"]:
        validate_time_boundary(period["signal_date"], period["return_date"])
        signal = np.asarray(period["signal"], dtype=float)
        future_return = np.asarray(period["future_return"], dtype=float)
        size = np.asarray(period["size"], dtype=float)
        industry = np.asarray(period["industry"], dtype=float)
        slopes.append(ols_slope(signal, future_return))
        correlations.append(correlation(signal, future_return))
        rank_correlations.append(correlation(ranks(signal), ranks(future_return)))
        spread_returns.append(
            float(future_return[np.argmax(signal)] - future_return[np.argmin(signal)])
        )
        _, exposure = neutralize(
            signal, size, industry, float(fixture["condition_limit"])
        )
        neutral_exposures.append(exposure)

    fm_mean = float(np.mean(slopes))
    fm_se = float(np.std(slopes, ddof=1) / math.sqrt(len(slopes)))
    mean_ic = float(np.mean(correlations))
    mean_rank_ic = float(np.mean(rank_correlations))
    neutral_max = max(neutral_exposures)
    gross = float(np.mean(spread_returns))
    net = gross - float(fixture["round_trip_cost"])
    large_capacity_net = net - float(fixture["large_capacity_extra_impact"])
    null_search = fixture["null_search"]
    p_values = null_search_p_values(
        seed=int(null_search["seed"]),
        observations=int(null_search["observations"]),
        attempts=int(null_search["attempts"]),
    )
    naive = sum(value < float(fixture["fdr_alpha"]) for value in p_values)
    bh = bh_rejections(p_values, float(fixture["fdr_alpha"]))

    alignment_rejected = 0
    try:
        validate_time_boundary("2025-03-31", "2025-03-31")
    except ValueError:
        alignment_rejected = 1
    unstable_rejected = 0
    try:
        neutralize(
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            float(fixture["condition_limit"]),
        )
    except ValueError:
        unstable_rejected = 1

    observed = {
        "fm_mean": fm_mean,
        "fm_se": fm_se,
        "ic": mean_ic,
        "rank_ic": mean_rank_ic,
        "neutral_max": neutral_max,
        "attempts": len(p_values),
        "naive": naive,
        "bh": bh,
        "gross": gross,
        "net": net,
        "large_capacity_net": large_capacity_net,
        "alignment_rejected": alignment_rejected,
        "unstable_rejected": unstable_rejected,
    }
    for name, expected in oracle["expected"].items():
        value = observed[name]
        if isinstance(expected, int):
            if value != expected:
                raise SystemExit(f"{name} ledger failed: observed={value} expected={expected}")
        elif abs(float(value) - float(expected)) > tolerance:
            raise SystemExit(f"{name} ledger failed: observed={value} expected={expected}")

    print(
        "oracle=passed "
        f"fm_mean={fm_mean:.6f} fm_se={fm_se:.6f} "
        f"ic={mean_ic:.6f} rank_ic={mean_rank_ic:.6f} "
        f"neutral_max={neutral_max:.6f} "
        f"tests=({len(p_values)},{naive},{bh}) "
        f"returns=({gross:.6f},{net:.6f},{large_capacity_net:.6f}) "
        f"windows={len(fixture['research_protocol']['windows'])} "
        f"decay_lags={len(fixture['research_protocol']['decay_lags_months'])} "
        "scope=global+a-share-boundary "
        f"alignment_rejected={alignment_rejected} unstable_rejected={unstable_rejected}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch01/oracle.json"))
    )
