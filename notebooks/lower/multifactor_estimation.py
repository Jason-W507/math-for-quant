# %% [markdown]
# # 多因子估计：透明算法与成熟库交叉验证
#
# **研究目标。** 从 OLS 过渡到 ridge/lasso，并区分数值稳定与经济识别。
# **假设。** 特征列已按研究协议处理；惩罚参数只在训练/选择窗口决定。
# **手算 oracle。** Ridge 解为 $(X^TX+\lambda I)^{-1}X^Ty$；lasso 的每个
# 坐标执行 soft-threshold 更新。

# %%
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.multifactor_estimation import (
    lasso_coordinate_descent,
    ridge_closed_form,
)
from math_for_quant.lower.multifactor_library import (
    cross_check_estimators,
    cross_check_route_statistics,
)
from math_for_quant.lower.multifactor import validate_time_boundary
from math_for_quant.lower.notebook_evidence import (
    assert_expected,
    load_oracle_and_fixture,
)


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    validate_time_boundary(fixture["feature_timestamp"], fixture["outcome_timestamp"])
    design = np.asarray(fixture["design"], dtype=float)
    target = np.asarray(fixture["target"], dtype=float)
    ridge_penalty = float(fixture["ridge_penalty"])
    lasso_penalty = float(fixture["lasso_penalty"])
    ridge = ridge_closed_form(design, target, ridge_penalty)
    lasso = lasso_coordinate_descent(design, target, lasso_penalty, 100_000)
    signals = np.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
    future = 0.01 + np.asarray([[0.02], [0.03]]) * signals
    factors = np.asarray([[-0.02], [0.01], [0.03], [-0.01], [0.02]])
    betas = np.asarray([0.5, 1.0, 1.5])
    assets = 0.004 + factors @ betas[None, :]
    library = cross_check_estimators(
        signals=signals,
        future_returns=future,
        asset_returns=assets,
        factor_returns=factors,
        regularization_design=design,
        regularization_target=target,
        ridge_penalty=ridge_penalty,
        lasso_penalty=lasso_penalty,
    )
    ridge_gap = float(np.max(np.abs(ridge - library.ridge_coefficients)))
    lasso_gap = float(np.max(np.abs(lasso - library.lasso_coefficients)))
    route_check = cross_check_route_statistics(
        panel_design=np.asarray([[0.0], [1.0], [0.0], [1.0], [1.0], [2.0]]),
        panel_target=np.asarray([1.0, 1.4, -2.0, -1.6, 0.9, 1.3]),
        entities=np.asarray([0, 0, 1, 1, 2, 2]),
        signal=np.asarray([-2.0, -1.0, 1.0, 2.0]),
        size=np.asarray([-1.0, -1.0, 1.0, 1.0]),
        industry=np.asarray([0.0, 1.0, 0.0, 1.0]),
        future_returns=np.asarray([-0.03, -0.01, 0.02, 0.04]),
        horizon_returns=np.asarray([
            [-0.03, -0.01, 0.02, 0.04],
            [-0.015, -0.004, 0.009, 0.02],
        ]),
        p_values=[0.001, 0.02, 0.3, 0.8],
        alpha=0.05,
    )
    route_gap = max(
        route_check.panel_slope_gap,
        route_check.neutralization_gap,
        route_check.ic_gap,
        route_check.rank_ic_gap,
        route_check.decay_gap,
        float(route_check.bh_count_gap),
    )

    # 图表与敏感性：比较同一系数在惩罚路径上的收缩。
    penalties = np.asarray([0.0, 0.1, 0.5, 1.0])
    path = [ridge_closed_form(design, target, value)[0] for value in penalties]
    plt.figure(figsize=(5, 2.5))
    plt.plot(penalties, path, marker="o")
    plt.close()

    # 故障注入：全零列必须被透明 lasso 拒绝。
    zero_column_rejected = 0
    try:
        lasso_coordinate_descent(np.zeros((4, 1)), target, 0.1)
    except ValueError:
        zero_column_rejected = 1
    observed = {
        "ridge": ridge[0],
        "lasso": lasso[0],
        "ridge_gap": ridge_gap,
        "lasso_gap": lasso_gap,
        "route_gap": route_gap,
        "zero_column_rejected": zero_column_rejected,
    }
    assert_expected(observed, oracle)
    print(
        f"estimation-oracle=passed ridge={ridge[0]:.6f} lasso={lasso[0]:.6f} "
        f"library_gaps=({ridge_gap:.3e},{lasso_gap:.3e}) route_gap={route_gap:.3e} "
        f"zero_column_rejected={zero_column_rejected}"
    )
    return 0


# %% [markdown]
# **限制。** 库一致性只证明实现对齐；惩罚参数是否具有样本外价值仍需滚动验证。

# %%
if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/multifactor-estimation/oracle.json")
    raise SystemExit(main(path))
