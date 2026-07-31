# %% [markdown]
# # 风险估计：协方差、风险贡献与估计误差
#
# **研究目标。** 从同一时间索引的收益矩阵构造协方差，比较线性收缩、因子模型与
# 风险平价，并用 bootstrap 报告组合波动率的估计不确定性。
#
# **手算 oracle。** 对角协方差的波动率为 0.2、0.3、0.4；逆波动率权重化简为
# `(6/13, 4/13, 3/13)`，每个资产贡献相同的组合方差。
#
# **假设。** 行按同一观测时点对齐；合成收益用于算法核验；普通 bootstrap 暂按
# 行 IID 重采样。若存在重叠收益或波动状态，必须更换重采样单位。

# %%
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture
from math_for_quant.lower.portfolio_estimation import validate_covariance
from math_for_quant.lower.portfolio_real_data import run_portfolio_real_data
from math_for_quant.lower.portfolio_route import run_estimation


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    observed = run_estimation(fixture)
    assert_expected(observed, oracle)
    assert_expected(observed, json.loads(Path(oracle["regression"]).read_text(encoding="utf-8")))
    sensitivity_fixture = dict(fixture)
    sensitivity_fixture["shrinkage_intensity"] = 0.50
    sensitivity = run_estimation(sensitivity_fixture)
    if np.isclose(sensitivity["shrunk_minimum_eigenvalue"], observed["shrunk_minimum_eigenvalue"]):
        raise SystemExit("shrinkage sensitivity did not change the covariance spectrum")
    real_data = run_portfolio_real_data(
        Path("data/real/stat-arb-us-macro-1999q4-2009q3.json")
    )
    if real_data["growth_rows"] != 39:
        raise SystemExit("portfolio real-data adapter did not preserve the time boundary")
    try:
        validate_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))
    except ValueError as error:
        if "positive semidefinite" not in str(error):
            raise
    else:
        raise SystemExit("invalid covariance was not rejected")
    plt.figure(figsize=(5, 2.5))
    plt.bar(["point", "lower", "upper"], [observed["bootstrap_volatility"], observed["bootstrap_lower"], observed["bootstrap_upper"]])
    plt.close()
    print(
        "portfolio-risk-estimation=passed "
        + " ".join(f"{key}={value:.6g}" for key, value in observed.items())
        + f" real_growth_rows={real_data['growth_rows']}"
        + f" real_covariance_trace={real_data['covariance_trace']:.6g}"
    )
    return 0


# %% [markdown]
# **失败边界。** 非半正定矩阵必须在优化前拒绝；bootstrap 默认逐行 IID 重采样，若
# 收益存在波动状态或共同冲击，应改为区块或分组重采样。收缩减少估计方差，不保证
# 目标矩阵正确，也不把宏观序列变成可交易资产收益。
#
# **敏感性实验。** 上面的可执行单元把收缩强度从 0.25 改为 0.50，并要求最小特征值
# 随之变化。读者还应比较 IID 与区块 bootstrap 的区间宽度，并记录重采样单位。

# %%
if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/portfolio-risk-estimation/oracle.json")))
