# %% [markdown]
# # 独立 oracle 最小实验
#
# 文本源是权威版本；`.ipynb` 是由 Jupytext 生成的出版产物。

# %%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main(
    oracle_path: Path = Path("evidence/foundation/oracle.json"),
) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
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


main()
