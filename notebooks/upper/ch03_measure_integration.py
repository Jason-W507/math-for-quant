# %% [markdown]
# # 简单函数积分与极限交换反例
#
# 解析结果来自独立手算；程序只复现有限求和和固定见证值。

# %%
from __future__ import annotations

import json
from pathlib import Path


def main(oracle_path: Path = Path("evidence/ch03/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    simple_integrals = []
    for level in oracle["simple_levels"]:
        bins = 2 ** int(level)
        simple_integrals.append(sum(index / bins / bins for index in range(bins)))

    spike_integrals = []
    point_values = []
    fixed_point = float(oracle["fixed_point"])
    for index in oracle["spike_indices"]:
        index = int(index)
        spike_integrals.append(index * (1.0 / index))
        point_values.append(float(index) if 0.0 < fixed_point <= 1.0 / index else 0.0)

    limit_integral = float(oracle["expected_limit_integral"])
    gap = spike_integrals[-1] - limit_integral
    observed = [*simple_integrals, *spike_integrals, *point_values, gap]
    expected = [
        *(float(item) for item in oracle["expected_simple_integrals"]),
        *(float(item) for item in oracle["expected_spike_integrals"]),
        *(float(item) for item in oracle["expected_point_values"]),
        float(oracle["expected_interchange_gap"]),
    ]
    tolerance = float(oracle["absolute_tolerance"])
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


main()
