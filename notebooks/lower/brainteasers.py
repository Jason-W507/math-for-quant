from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def reachable_by_adjacent_flips(target: list[int]) -> bool:
    state = [0] * len(target)
    for index in range(len(target) - 1):
        if state[index] != target[index]:
            state[index] ^= 1
            state[index + 1] ^= 1
    return state == target


def recover_missing(n: int, missing_sum: int, missing_square_sum: int) -> tuple[int, int]:
    product_numerator = missing_sum * missing_sum - missing_square_sum
    if product_numerator % 2 != 0:
        raise ValueError("inconsistent missing-number ledger")
    product = product_numerator // 2
    discriminant = missing_sum * missing_sum - 4 * product
    if discriminant < 0:
        raise ValueError("inconsistent missing-number ledger")
    root = math.isqrt(discriminant)
    if root * root != discriminant:
        raise ValueError("inconsistent missing-number ledger")
    if (missing_sum - root) % 2 != 0 or (missing_sum + root) % 2 != 0:
        raise ValueError("inconsistent missing-number ledger")
    values = ((missing_sum - root) // 2, (missing_sum + root) // 2)
    if values[0] == values[1] or not (1 <= values[0] < values[1] <= n):
        raise ValueError("missing numbers outside the declared universe")
    if sum(values) != missing_sum or sum(value * value for value in values) != missing_square_sum:
        raise ValueError("inconsistent missing-number ledger")
    return values


def main(path: str) -> int:
    oracle_path = Path(path)
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root / oracle["fixture"]["path"]).read_text(encoding="utf-8"))
    reachable = reachable_by_adjacent_flips(data["light_switch_reachable"])
    missing = recover_missing(data["missing_n"], data["missing_sum"], data["missing_square_sum"])
    expected = tuple(oracle["expected"])
    if reachable != oracle["expected_reachable"] or missing != expected:
        raise AssertionError(f"oracle mismatch: reachable={reachable} missing={missing}")
    print(f"brainteasers=passed reachable={int(reachable)} missing={missing[0]},{missing[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
