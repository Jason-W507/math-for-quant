# %% [markdown]
# # Lp 范数、求积误差与 Fubini 反例
#
# 解析范数、复合中点余项和双重级数计数先由手算给出；程序只复现这些 oracle。

# %%
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def coefficient(row: int, column: int) -> int:
    if row == column:
        return 1
    if row == column + 1:
        return -1
    return 0


def row_sum(row: int) -> int:
    first_column = max(1, row - 1)
    return sum(coefficient(row, column) for column in range(first_column, row + 1))


def column_sum(column: int) -> int:
    return sum(coefficient(row, column) for row in (column, column + 1))


def main(oracle_path: Path = Path("evidence/ch04/oracle.json")) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if oracle["p_values"] != [1, 2, 4]:
        raise SystemExit("ledger gate failed: p values must equal [1, 2, 4]")
    if oracle["midpoint_bins"] != 64:
        raise SystemExit("ledger gate failed: midpoint bins must equal 64")
    if oracle["section_sizes"] != [10, 100]:
        raise SystemExit(
            "ledger gate failed: section sizes must equal [10, 100]"
        )

    section_count = len(oracle["section_sizes"])
    if (
        len(oracle["expected_lp_norms"]) != len(oracle["p_values"])
        or len(oracle["expected_square_sums"]) != section_count
        or len(oracle["expected_rectangular_sums"]) != section_count
        or len(oracle["expected_absolute_square_sums"]) != section_count
    ):
        raise SystemExit(
            "ledger gate failed: expected counts must match input counts"
        )

    numeric_scalars = [
        *oracle["expected_lp_norms"],
        oracle["expected_midpoint_integral"],
        oracle["expected_midpoint_error"],
        oracle["expected_midpoint_bound"],
        oracle["expected_row_first"],
        oracle["expected_column_first"],
        *oracle["expected_square_sums"],
        *oracle["expected_rectangular_sums"],
        *oracle["expected_absolute_square_sums"],
        oracle["expected"],
        oracle["absolute_tolerance"],
    ]
    try:
        parsed_scalars = [float(value) for value in numeric_scalars]
    except (TypeError, ValueError):
        raise SystemExit(
            "numeric gate failed: oracle scalars must be finite numbers"
        ) from None
    if not all(math.isfinite(value) for value in parsed_scalars):
        raise SystemExit("numeric gate failed: oracle scalars must be finite")
    tolerance = float(oracle["absolute_tolerance"])
    if tolerance != 1e-12:
        raise SystemExit(
            "ledger gate failed: absolute tolerance must equal 1e-12"
        )
    if (
        oracle["expected_row_first"] != 1
        or oracle["expected_column_first"] != 0
        or oracle["expected_square_sums"] != [1, 1]
        or oracle["expected_rectangular_sums"] != [0, 0]
        or oracle["expected_absolute_square_sums"] != [19, 199]
    ):
        raise SystemExit(
            "ledger gate failed: iterated-sum labels must match analytic ledger"
        )
    if not math.isclose(
        float(oracle["expected"]),
        1.0 / math.sqrt(3.0),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise SystemExit(
            "ledger gate failed: published expected must equal analytic L2 norm"
        )

    lp_norms = [
        (1.0 / (int(power) + 1)) ** (1.0 / int(power))
        for power in oracle["p_values"]
    ]

    bins = int(oracle["midpoint_bins"])
    midpoint_integral = sum(((index + 0.5) / bins) ** 2 for index in range(bins)) / bins
    midpoint_error = (1.0 / 3.0) - midpoint_integral
    midpoint_bound = 2.0 / (24.0 * bins * bins)

    square_sums = []
    rectangular_sums = []
    absolute_square_sums = []
    for size_value in oracle["section_sizes"]:
        size = int(size_value)
        square_sums.append(
            sum(coefficient(row, column) for row in range(1, size + 1) for column in range(1, size + 1))
        )
        rectangular_sums.append(
            sum(coefficient(row, column) for row in range(1, size + 2) for column in range(1, size + 1))
        )
        absolute_square_sums.append(
            sum(abs(coefficient(row, column)) for row in range(1, size + 1) for column in range(1, size + 1))
        )

    support_check_size = max(int(item) for item in oracle["section_sizes"])
    row_sums = [row_sum(row) for row in range(1, support_check_size + 1)]
    column_sums = [
        column_sum(column) for column in range(1, support_check_size + 1)
    ]
    if any(value != 0 for value in row_sums[1:]) or any(
        value != 0 for value in column_sums
    ):
        raise SystemExit(
            f"unexpected iterated supports: rows={row_sums} columns={column_sums}"
        )
    row_first = sum(row_sums)
    column_first = sum(column_sums)
    observed = [
        *lp_norms,
        midpoint_integral,
        midpoint_error,
        midpoint_bound,
        row_first,
        column_first,
        *square_sums,
        *rectangular_sums,
        *absolute_square_sums,
    ]
    expected = [
        *(float(item) for item in oracle["expected_lp_norms"]),
        float(oracle["expected_midpoint_integral"]),
        float(oracle["expected_midpoint_error"]),
        float(oracle["expected_midpoint_bound"]),
        int(oracle["expected_row_first"]),
        int(oracle["expected_column_first"]),
        *(int(item) for item in oracle["expected_square_sums"]),
        *(int(item) for item in oracle["expected_rectangular_sums"]),
        *(int(item) for item in oracle["expected_absolute_square_sums"]),
    ]
    if len(observed) != len(expected) or any(
        abs(left - right) > tolerance for left, right in zip(observed, expected)
    ):
        raise SystemExit(f"oracle mismatch: observed={observed} expected={expected}")

    print(
        "oracle=passed "
        f"lp1={lp_norms[0]:.6f} lp2={lp_norms[1]:.6f} lp4={lp_norms[2]:.6f} "
        f"midpoint={midpoint_integral:.9f} error={midpoint_error:.9f} "
        f"bound={midpoint_bound:.9f} row_first={row_first} col_first={column_first} "
        f"square10={square_sums[0]} rectangle10={rectangular_sums[0]} abs10={absolute_square_sums[0]} "
        f"square100={square_sums[1]} rectangle100={rectangular_sums[1]} abs100={absolute_square_sums[1]}"
    )
    return 0


oracle_path = Path("evidence/ch04/oracle.json")
if Path(sys.argv[0]).stem == "ch04_lp_product_measure" and len(sys.argv) > 1:
    oracle_path = Path(sys.argv[1])
main(oracle_path)
