# %% [markdown]
# # 随机分析与无套利：从二次变差到测度变换
#
# **研究目标。** 用同一路径的嵌套分割观察二次变差，并区分 Novikov 充分条件与
# 教材自定义能量预算。
# **假设。** 细网格增量来自同一条标准 Brownian 路径；被积过程适应且平方可积；
# Girsanov 密度严格为正、归一化且满足使其成为真鞅的充分条件。
# **手算 oracle。** 离散平方恒等式精确成立；`mu - sigma*theta = r`。
# **失败注入。** `theta_t=(T-t)^(-1/2)` 在完整端点的平方积分发散。

# %%
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

from math_for_quant.lower.derivatives_route import run_stochastic
from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    observed = run_stochastic(fixture)
    assert_expected(observed, oracle)
    regression = json.loads(Path(oracle["regression"]).read_text(encoding="utf-8"))
    assert_expected(observed, regression)
    plt.figure(figsize=(5, 2.5))
    plt.plot([64, 16, 1], [observed["qv_coarse"], observed["qv_medium"], observed["qv_fine"]], marker="o")
    plt.axhline(1.0, color="black", linestyle="--")
    plt.close()
    print("derivatives-stochastic=passed " + " ".join(f"{key}={value:.6g}" for key, value in observed.items()))
    return 0


# %% [markdown]
# **限制。** 单条模拟路径只演示分割细化，不证明依概率收敛；理论结论仍依赖定理条件。

# %%
if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/derivatives-stochastic/oracle.json")))
