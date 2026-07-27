# %% [markdown]
# # 简单函数积分与极限交换反例
#
# 解析结果来自独立手算；程序只复现有限求和和固定见证值。

# %%
from __future__ import annotations

import json
import math
from pathlib import Path
import sys


def main(oracle_path: Path = Path("evidence/ch03/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    try:
        fixed_point = float(oracle["fixed_point"])
        expected_simple_integrals = [
            float(item) for item in oracle["expected_simple_integrals"]
        ]
        expected_spike_integrals = [
            float(item) for item in oracle["expected_spike_integrals"]
        ]
        expected_point_values = [
            float(item) for item in oracle["expected_point_values"]
        ]
        limit_integral = float(oracle["expected_limit_integral"])
        expected_gap = float(oracle["expected_interchange_gap"])
        tolerance = float(oracle["absolute_tolerance"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise SystemExit("numeric gate failed: oracle scalars must be finite")
    numeric_scalars = [
        fixed_point,
        *expected_simple_integrals,
        *expected_spike_integrals,
        *expected_point_values,
        limit_integral,
        expected_gap,
        tolerance,
    ]
    if not all(math.isfinite(item) for item in numeric_scalars):
        raise SystemExit("numeric gate failed: oracle scalars must be finite")
    if limit_integral != 0.0:
        raise SystemExit("ledger gate failed: limit integral must equal 0")
    if fixed_point != 0.2:
        raise SystemExit("ledger gate failed: fixed point must equal 0.2")
    simple_levels = oracle["simple_levels"]
    spike_indices = oracle["spike_indices"]
    if simple_levels != [2, 4, 8]:
        raise SystemExit("ledger gate failed: simple levels must equal [2, 4, 8]")
    if spike_indices != [10, 100]:
        raise SystemExit("ledger gate failed: spike indices must equal [10, 100]")
    if (
        len(expected_simple_integrals) != len(simple_levels)
        or len(expected_spike_integrals) != len(spike_indices)
        or len(expected_point_values) != len(spike_indices)
    ):
        raise SystemExit("ledger gate failed: expected counts must match input counts")

    simple_integrals = []
    for level in simple_levels:
        bins = 2 ** int(level)
        simple_integrals.append(sum(index / bins / bins for index in range(bins)))

    spike_integrals = []
    point_values = []
    for index in spike_indices:
        index = int(index)
        spike_integrals.append(index * (1.0 / index))
        point_values.append(float(index) if 0.0 < fixed_point <= 1.0 / index else 0.0)

    gap = spike_integrals[-1] - limit_integral
    observed = [*simple_integrals, *spike_integrals, *point_values, gap]
    expected = [
        *expected_simple_integrals,
        *expected_spike_integrals,
        *expected_point_values,
        expected_gap,
    ]
    if any(abs(left - right) > tolerance for left, right in zip(observed, expected)):
        raise SystemExit(f"oracle mismatch: observed={observed} expected={expected}")

    print(
        "oracle=passed "
        f"simple_m2={simple_integrals[0]:.6f} "
        f"simple_m4={simple_integrals[1]:.6f} "
        f"simple_m8={simple_integrals[2]:.6f} "
        f"spike10={spike_integrals[0]:.6f} "
        f"spike100={spike_integrals[1]:.6f} "
        f"point10={point_values[0]:.6f} point100={point_values[1]:.6f} "
        f"gap={gap:.6f}"
    )
    return 0


oracle_path = Path("evidence/ch03/oracle.json")
if Path(sys.argv[0]).stem == "ch03_measure_integration" and len(sys.argv) > 1:
    oracle_path = Path(sys.argv[1])
main(oracle_path)
