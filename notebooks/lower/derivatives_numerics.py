# %% [markdown]
# # 定价与校准：闭式、树、PDE、Monte Carlo 与曲面
#
# **研究目标。** 对同一合约分开报告离散、截断和采样误差，并把逐点隐波反演与
# 参数化总方差曲面校准分成两步。
# **失败注入。** 非等距执行价必须比较相邻斜率，不能套等距二阶差分。

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

from math_for_quant.lower.derivatives_route import run_numerics
from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    observed = run_numerics(fixture)
    assert_expected(observed, oracle)
    regression = json.loads(Path(oracle["regression"]).read_text(encoding="utf-8"))
    assert_expected(observed, regression)
    plt.figure(figsize=(5, 2.5))
    plt.bar(["tree", "PDE", "MC"], [observed["tree_error"], observed["pde_error"], observed["mc_error"]])
    plt.close()
    print("derivatives-numerics=passed " + " ".join(f"{key}={value:.6g}" for key, value in observed.items()))
    return 0


# %% [markdown]
# **限制。** 合成曲面由模型自身生成，只能验证恢复与约束；真实报价还需要 bid/ask、
# 报价时间、远期和贴现曲线。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/derivatives-numerics/oracle.json")))
