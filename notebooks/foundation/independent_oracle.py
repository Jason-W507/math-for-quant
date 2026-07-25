# %% [markdown]
# # 独立 oracle 最小实验
#
# 文本源是权威版本；`.ipynb` 是由 Jupytext 生成的出版产物。

# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    returns = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    observed = float(np.mean(returns))
    expected = float(oracle["expected"])
    tolerance = float(oracle["absolute_tolerance"])

    if abs(observed - expected) > tolerance:
        raise SystemExit(
            f"oracle mismatch: observed={observed:.12f} expected={expected:.12f}"
        )

    print(
        f"oracle=passed observed={observed:.12f} expected={expected:.12f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
