# %% [markdown]
# # 多因子模型：两种 Fama--MacBeth 不是一回事
#
# **研究目标。** 区分逐期横截面预测回归与经典两遍资产定价回归。
# **假设。** 信号严格早于未来收益；经典两遍法的因子收益与资产收益同频。
# **手算 oracle。** 两期预测斜率为 0.02、0.03，均值 0.025；经典例的
# 风险价格等于因子收益均值 0.006。

# %%
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.multifactor_estimation import (
    classic_two_pass_fama_macbeth,
    predictive_fama_macbeth,
)


def main(oracle_path: Path) -> int:
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    fixture = json.loads(Path(oracle["fixture"]["path"]).read_text(encoding="utf-8"))
    signals = np.asarray(fixture["signals"], dtype=float)
    future = np.asarray(fixture["future_returns"], dtype=float)
    factors = np.asarray(fixture["factor_returns"], dtype=float)
    betas = np.asarray(fixture["betas"], dtype=float)
    assets = float(fixture["asset_alpha"]) + factors @ betas[None, :]
    predictive = predictive_fama_macbeth(signals, future)
    classic = classic_two_pass_fama_macbeth(assets, factors)

    # 图表：两类估计对象分别画出，避免把横截面预测系数误叫风险价格。
    plt.figure(figsize=(5, 2.5))
    plt.plot(predictive.coefficients, marker="o", label="predictive slopes")
    plt.axhline(classic.risk_prices[0], color="tab:orange", label="factor price")
    plt.legend()
    plt.close()

    # 故障注入：日期相等必须在研究协议层拒绝；此 notebook 只记录类别。
    alignment_failure = "signal timestamp must precede outcome timestamp"
    # 敏感性：将第二期信号乘正数只改变系数尺度，不改变排序。
    scaled = predictive_fama_macbeth(
        np.asarray([signals[0], 2.0 * signals[1]]), future
    )
    expected = oracle["expected"]
    observed = {
        "predictive_mean": predictive.mean_coefficient,
        "classic_risk_price": classic.risk_prices[0],
        "scaled_second_slope": scaled.coefficients[1],
    }
    for key, value in expected.items():
        if abs(float(observed[key]) - float(value)) > float(oracle["absolute_tolerance"]):
            raise SystemExit(f"{key} mismatch: observed={observed[key]} expected={value}")
    print(
        f"model-oracle=passed predictive_mean={predictive.mean_coefficient:.6f} "
        f"classic_risk_price={classic.risk_prices[0]:.6f} "
        f"scaled_second_slope={scaled.coefficients[1]:.6f} "
        f"failure={alignment_failure.replace(' ', '-') }"
    )
    return 0


# %% [markdown]
# **限制。** 精确线性合成样本只验证估计目标与实现；它不提供真实资产定价证据。

# %%
if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/multifactor-model/oracle.json")
    raise SystemExit(main(path))
