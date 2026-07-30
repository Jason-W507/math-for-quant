# %% [markdown]
# # 动态与长期关系：从单位根到 OU
#
# **研究目标。** 在同一条合成路径上区分水平回归、残差单位根、
# Engle--Granger、Johansen、ECM 与 OU 映射。
# **假设。** 两条水平序列共享一条固定长期关系，残差服从平稳 AR(1)，
# 采样间隔固定；OU 映射只在估计自回归系数落于 `(0, 1)` 时成立。
# **手算 oracle。** 长期斜率接近 2，残差按 0.55 衰减。
# **敏感性。** 固定过程参数，将起点偏离从 `0.25` 提高到 `1.0`；
# 命中均值的期望时间必须增加。

# %%
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture
from math_for_quant.lower.stat_arb_models import engle_granger, fit_ecm, johansen_rank, ou_diagnostics


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    count = int(fixture["observation_count"])
    increments = np.resize(np.asarray(fixture["x_increments"], dtype=float), count)
    x = np.cumsum(increments)
    residual = float(fixture["residual_phi"]) ** np.arange(count)
    y = float(fixture["cointegration_intercept"]) + float(fixture["cointegration_slope"]) * x + residual
    relation = engle_granger(y, x)
    ecm = fit_ecm(y, x, relation)
    ou = ou_diagnostics(relation.residuals, step=float(fixture["step"]))
    near_start = ou_diagnostics(
        relation.residuals,
        step=float(fixture["step"]),
        starting_displacement=0.25,
    )
    far_start = ou_diagnostics(
        relation.residuals,
        step=float(fixture["step"]),
        starting_displacement=1.0,
    )
    plt.figure(figsize=(5, 2.5)); plt.plot(relation.residuals); plt.axhline(0.0, color="black"); plt.close()
    rejected = 0
    try:
        ou_diagnostics(2.0 ** np.arange(8.0), step=1.0)
    except ValueError:
        rejected = 1
    observed = {
        "slope": relation.slope,
        "adf_statistic": relation.residual_adf_statistic,
        "johansen_rank": johansen_rank(np.column_stack([y, x])),
        "ecm_speed": ecm.adjustment_speed,
        "half_life": ou.half_life,
        "expected_first_passage": ou.expected_first_passage,
        "nonstationary_ou_rejected": rejected,
        "first_passage_displacement_sensitivity": (
            far_start.expected_first_passage - near_start.expected_first_passage
        ),
    }
    assert_expected(observed, oracle)
    print("stat-arb-model=passed " + " ".join(f"{key}={value:.6f}" for key, value in observed.items()))
    return 0


# %% [markdown]
# **限制。** 小样本临界值依赖确定性项和滞后选择；真实研究必须报告库检验表，不能把单个 t 值当成定理。

# %%
if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/stat-arb-model/oracle.json")
    raise SystemExit(main(path))
