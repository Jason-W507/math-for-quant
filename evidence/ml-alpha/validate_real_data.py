from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def independent_held_out_mse(rows: list[dict[str, object]]) -> float:
    train, test = rows[:5], rows[5:]
    x = [float(row["signal"]) for row in train]
    y = [float(row["outcome"]) for row in train]
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y, strict=True)
    ) / denominator
    intercept = y_mean - slope * x_mean
    errors = [
        intercept + slope * float(row["signal"]) - float(row["outcome"])
        for row in test
    ]
    return sum(error**2 for error in errors) / len(errors)


def main(path: Path) -> int:
    oracle = json.loads(path.read_text(encoding="utf-8"))
    fixture_path = Path(oracle["fixture"]["path"])
    if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != oracle["fixture"]["sha256"]:
        raise SystemExit("ML real-data snapshot hash mismatch")
    snapshot = json.loads(fixture_path.read_text(encoding="utf-8"))
    if snapshot["signal_year"] >= snapshot["outcome_year"]:
        raise SystemExit("ML real-data timestamp protocol rejected")
    rows = snapshot["rows"]
    observed = {
        "rows": len(rows),
        "train_rows": 5,
        "test_rows": len(rows) - 5,
        "test_mse": independent_held_out_mse(rows),
    }
    tolerance = float(oracle["absolute_tolerance"])
    for key, expected in oracle["expected"].items():
        if abs(float(observed[key]) - float(expected)) > tolerance:
            raise SystemExit(f"ML real-data {key} mismatch")
    print(
        "ml-real-data-oracle=passed "
        f"rows={observed['rows']} train={observed['train_rows']} "
        f"test={observed['test_rows']} test_mse={observed['test_mse']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/ml-alpha/real-data-oracle.json")))
