# %% [markdown]
# # 对冲失效与研究包：从单路径误差到分布
#
# **研究目标。** 在固定多路径样本上报告 bias、RMSE、分位数和成本分布，而不是
# 用一条路径代表对冲质量。
# **双实现。** 透明逐路径账本与向量化实现必须逐路径一致。

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

from math_for_quant.lower.derivatives_route import run_hedging
from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    observed = run_hedging(fixture)
    assert_expected(observed, oracle)
    regression = json.loads(Path(oracle["regression"]).read_text(encoding="utf-8"))
    assert_expected(observed, regression)
    plt.figure(figsize=(5, 2.5))
    plt.bar(["q05", "q50", "q95"], [observed["error_q05"], observed["error_q50"], observed["error_q95"]])
    plt.close()
    print("derivatives-hedging=passed " + " ".join(f"{key}={value:.6g}" for key, value in observed.items()))
    return 0


# %% [markdown]
# **限制。** GBM、固定波动率和比例成本不含跳跃、波动率微笑、冲击或融资约束。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/derivatives-hedging/oracle.json")))
