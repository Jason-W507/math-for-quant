# %% [markdown]
# # 压缩迭代与非一致收敛反例
#
# Jupytext 文本源只复现独立手算账本；一般收敛结论仍由正文证明。

# %%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main(oracle_path: Path = Path("evidence/ch02/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    value = float(oracle["initial_value"])
    intercept = float(oracle["intercept"])
    factor = float(oracle["contraction_factor"])
    iterations = int(oracle["iterations"])
    first_step = intercept + factor * value
    for _ in range(iterations):
        value = intercept + factor * value

    fixed_point = float(oracle["expected_fixed_point"])
    error = abs(fixed_point - value)
    bound = factor**iterations / (1.0 - factor) * abs(first_step - float(oracle["initial_value"]))
    witness_errors = []
    for index in oracle["witness_indices"]:
        witness = np.exp(-np.log(2.0) / int(index))
        witness_errors.append(float(witness ** int(index)))

    observed = [value, error, bound, *witness_errors]
    expected = [
        float(oracle["expected_iterate"]),
        float(oracle["expected_error"]),
        float(oracle["expected_bound"]),
        *(float(item) for item in oracle["expected_witness_errors"]),
    ]
    tolerance = float(oracle["absolute_tolerance"])
    if any(abs(left - right) > tolerance for left, right in zip(observed, expected)):
        raise SystemExit(f"oracle mismatch: observed={observed} expected={expected}")

    print(
        "oracle=passed "
        f"contraction_x5={value:.6f} error={error:.6f} bound={bound:.6f} "
        f"witness10={witness_errors[0]:.6f} witness100={witness_errors[1]:.6f}"
    )
    return 0


main()
