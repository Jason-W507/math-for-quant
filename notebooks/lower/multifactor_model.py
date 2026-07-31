# %% [markdown]
# # 多因子模型：两种 Fama--MacBeth 不是一回事
#
# **研究目标。** 区分逐期横截面预测回归与经典两遍资产定价回归。
# **假设。** 信号严格早于未来收益；经典两遍法的因子收益与资产收益同频。
# **手算 oracle。** 两期预测斜率为 0.02、0.03，均值 0.025；经典例的
# 风险价格等于因子收益均值 0.006。

# %%
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.multifactor_estimation import (
    classic_two_pass_fama_macbeth,
    predictive_fama_macbeth,
)
from math_for_quant.lower.multifactor import validate_time_boundary
from math_for_quant.lower.notebook_evidence import (
    assert_expected,
    load_oracle_and_fixture,
)


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    for signal_date, return_date in zip(
        fixture["signal_dates"], fixture["return_dates"], strict=True
    ):
        validate_time_boundary(signal_date, return_date)
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

    # 故障注入：日期相等必须由真实验证器拒绝。
    alignment_rejected = 0
    try:
        validate_time_boundary("2025-03-31", "2025-03-31")
    except ValueError:
        alignment_rejected = 1
    # 敏感性：将第二期信号乘正数只改变系数尺度，不改变排序。
    scaled = predictive_fama_macbeth(
        np.asarray([signals[0], 2.0 * signals[1]]), future
    )
    expected = oracle["expected"]
    observed = {
        "predictive_mean": predictive.mean_coefficient,
        "classic_risk_price": classic.risk_prices[0],
        "scaled_second_slope": scaled.coefficients[1],
        "alignment_rejected": alignment_rejected,
    }
    assert_expected(observed, oracle)
    print(
        f"multifactor-model=passed predictive_mean={predictive.mean_coefficient:.6f} "
        f"classic_risk_price={classic.risk_prices[0]:.6f} "
        f"scaled_second_slope={scaled.coefficients[1]:.6f} "
        f"alignment_rejected={alignment_rejected}"
    )
    return 0


# %% [markdown]
# **限制。** 精确线性合成样本只验证估计目标与实现；它不提供真实资产定价证据。

# %%
if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/multifactor-model/oracle.json")
    raise SystemExit(main(path))
