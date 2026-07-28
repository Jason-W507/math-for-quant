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
    order = np.argsort(values, kind="stable")
    result = np.empty(values.size, dtype=float)
    result[order] = np.arange(1, values.size + 1, dtype=float)
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
    p_values = [float(value) for value in fixture["null_p_values"]]
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
        f"alignment_rejected={alignment_rejected} unstable_rejected={unstable_rejected}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/lower-ch01/oracle.json"))
    )
