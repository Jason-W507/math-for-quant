# %% [markdown]
# # 组合优化：Black--Litterman、CVaR 与稳健实施
#
# **研究目标。** 把先验、观点与置信度写进 Black--Litterman 后验；把 CVaR 写成
# Rockafellar--Uryasev 线性规划；最后加入期望收益不确定性、换手、成本和可交易性。
#
# **手算 oracle。** 两资产 CVaR LP 的最优权重为 `(0.2, 0.8)`，最坏四分之一的平均
# 损失为 0.013；网格枚举必须重建同一答案。0.8 双边换手乘十万元与 10bp 等于 80。
#
# **假设。** 场景在决策时已冻结；权重非负且满仓；不可交易标记是硬约束；成本率按
# 双边换手计提。Black--Litterman 观点只表达条件输入，不是真实收益标签。

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture
from math_for_quant.lower.portfolio_optimization import robust_cost_aware_rebalance
from math_for_quant.lower.portfolio_route import run_optimization


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    observed = run_optimization(fixture)
    assert_expected(observed, oracle)
    sensitivity_fixture = dict(fixture)
    sensitivity_fixture["risk_aversion"] = 3.0
    sensitivity = run_optimization(sensitivity_fixture)
    if np.isclose(sensitivity["robust_score"], observed["robust_score"]):
        raise SystemExit("risk-aversion sensitivity did not change the robust objective")
    try:
        robust_cost_aware_rebalance(
            np.array([0.08, 0.04]), np.eye(2), np.array([0.5, 0.5]), np.array([0.01, 0.01]),
            risk_aversion=0.0, uncertainty_penalty=1.0, cost_rate=0.0, capital=1.0,
            maximum_weight=1.0, tradable=np.array([1, 1]), grid_step=0.1,
        )
    except ValueError as error:
        if "risk aversion" not in str(error):
            raise
    else:
        raise SystemExit("invalid risk-aversion contract was not rejected")
    plt.figure(figsize=(4, 2.5))
    plt.bar(["CVaR-1", "CVaR-2", "robust-1", "robust-2"], [observed["cvar_weight_1"], observed["cvar_weight_2"], observed["rebalance_weight_1"], observed["rebalance_weight_2"]])
    plt.close()
    print("portfolio-risk-optimization=passed " + " ".join(f"{key}={value:.6g}" for key, value in observed.items()))
    return 0


# %% [markdown]
# **失败边界。** 后验不是“客观收益”，观点协方差编码置信度。CVaR 的场景矩阵必须
# 冻结时间边界；不可交易资产是硬约束，不能用更高成本代替。任何解都要重算权重和、
# 目标值、活跃边界、换手和现金成本。
#
# **敏感性实验。** 上面的可执行单元把风险厌恶从 1 提高到 3，并要求稳健目标变化。
# 实务报告还应扫描观点置信度、最大权重、成本率和不可交易集合，而不是只展示一个解。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/portfolio-risk-optimization/oracle.json")))
