# %% [markdown]
# # 尾部与压力风险：有效样本、非线性重估和反向压力
#
# **研究目标。** 让 VaR/ES 的输出携带有效尾部观察数、离散分位误差与 bootstrap
# 区间，并把历史尾部、给定压力和反向压力分成三种问题。
#
# **手算 oracle。** 500 个等距损失在 95% 水平只有 25 个有效尾部观察，因此落在
# `warn` 而非 `pass`；二阶重估的单位冲击损失为 4.016，达到损失 10 的最小尺度为 3.125。
#
# **假设。** 损失已按同一估值时点和单位整理；历史尾部只描述该样本；Delta--Gamma
# 是局部二阶近似；反向压力方向由研究者预先声明，不能由结果反推概率。

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture
from math_for_quant.lower.portfolio_route import run_tail
from math_for_quant.lower.portfolio_real_data import run_portfolio_real_data
from math_for_quant.lower.portfolio_tail import empirical_tail_risk


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    observed = run_tail(fixture)
    assert_expected(observed, oracle)
    assert_expected(observed, json.loads(Path(oracle["regression"]).read_text(encoding="utf-8")))
    sensitivity_fixture = dict(fixture)
    sensitivity_fixture["confidence"] = 0.90
    sensitivity_fixture["warning_tail_observations"] = 60
    sensitivity = run_tail(sensitivity_fixture)
    if np.isclose(sensitivity["value_at_risk"], observed["value_at_risk"]):
        raise SystemExit("confidence sensitivity did not change VaR")
    real_data = run_portfolio_real_data(
        Path("data/real/stat-arb-us-macro-1999q4-2009q3.json")
    )
    if real_data["tail_status"] != "reject":
        raise SystemExit("real-data tail gate must reject fewer than 20 effective observations")
    try:
        empirical_tail_risk(np.arange(100.0), 0.99, minimum_tail_observations=20)
    except ValueError as error:
        if "effective tail observations" not in str(error):
            raise
    else:
        raise SystemExit("insufficient tail sample was not rejected")
    plt.figure(figsize=(4, 2.5))
    plt.bar(["linear", "nonlinear", "threshold"], [observed["linear_unit_loss"], observed["nonlinear_unit_loss"], observed["reverse_stress_loss"]])
    plt.close()
    print(
        "portfolio-risk-tail=passed "
        + " ".join(
            f"{key}={value}" if isinstance(value, str) else f"{key}={value:.6g}"
            for key, value in observed.items()
        )
        + f" real_tail_status={real_data['tail_status']}"
    )
    return 0


# %% [markdown]
# **限制。** IID bootstrap 不适合波动聚集或重叠收益；二阶重估不覆盖跳跃、隐含曲面
# 重标定与流动性反馈。反向压力只回答“沿给定方向多大才触发阈值”，不赋予该方向概率。
#
# **敏感性实验。** 上面的可执行单元把置信水平从 95\% 改为 90\%，同时重算有效尾部
# 数、状态与 VaR。读者还应扫描压力方向、阈值和 Gamma，并检查第一处阈值穿越。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/portfolio-risk-tail/oracle.json")))
