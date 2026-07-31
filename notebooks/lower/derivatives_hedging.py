# %% [markdown]
# # 对冲失效与研究包：从单路径误差到分布
#
# **研究目标。** 在固定多路径样本上报告 bias、RMSE、分位数和成本分布，而不是
# 用一条路径代表对冲质量。
#
# **假设。** 风险中性 GBM、常波动率、离散调仓、比例成本、无冲击和无限流动性。
# **手算 oracle。** 初始股票交易成本为 `c*abs(Delta_0)*S_0`，随后现金账户先计息
# 再调仓；到期误差必须从股票、现金与 payoff 同一账本得到。
# **双实现。** 透明逐路径账本与 SciPy `ndtr` 驱动的向量化实现必须逐路径一致。
# **失败注入。** 负成本率必须以稳定诊断拒绝，不能作为“返佣”静默进入结果。

# %%
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt

from math_for_quant.lower.derivatives import call_delta
from math_for_quant.lower.derivatives_hedging import greek_convergence
from math_for_quant.lower.derivatives_route import run_hedging
from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    observed = run_hedging(fixture)
    assert_expected(observed, oracle)
    regression = json.loads(Path(oracle["regression"]).read_text(encoding="utf-8"))
    assert_expected(observed, regression)
    hand_initial_cost = (
        float(fixture["cost_rate"])
        * abs(call_delta(
            float(fixture["spot"]), float(fixture["strike"]),
            float(fixture["rate"]), float(fixture["sigma"]),
            float(fixture["maturity"]),
        ))
        * float(fixture["spot"])
    )
    financed_hand_cost = hand_initial_cost * math.exp(
        float(fixture["rate"]) * float(fixture["maturity"])
    )
    if abs(hand_initial_cost - observed["one_step_raw_cost"]) > 1e-12:
        raise SystemExit("one-step raw cost disagrees with the hand oracle")
    if abs(financed_hand_cost - observed["one_step_financed_drag"]) > 1e-12:
        raise SystemExit("one-step financed cost disagrees with the hand oracle")
    if observed["one_step_cost_identity_gap"] > 1e-12:
        raise SystemExit("one-step self-financing identity failed")
    if hand_initial_cost <= 0.0 or observed["negative_cost_rejected"] != 1:
        raise SystemExit("hedging hand oracle or failure injection did not pass")
    plt.figure(figsize=(5, 2.5))
    plt.plot(
        [int(fixture["coarse_steps"]), int(fixture["steps"]), int(fixture["fine_steps"])],
        [
            observed["coarse_after_cost_rmse"], observed["after_cost_rmse"],
            observed["fine_after_cost_rmse"],
        ],
        marker="o",
        label="after-cost RMSE",
    )
    plt.plot(
        [int(fixture["coarse_steps"]), int(fixture["steps"]), int(fixture["fine_steps"])],
        [observed["coarse_mean_cost"], observed["mean_cost"], observed["fine_mean_cost"]],
        marker="s",
        label="mean cost",
    )
    plt.legend()
    plt.close()
    greek_errors = greek_convergence(
        float(fixture["spot"]), float(fixture["strike"]),
        float(fixture["rate"]), float(fixture["sigma"]),
        float(fixture["maturity"]),
        steps=tuple(float(value) for value in fixture["greek_steps"]),
    )
    plt.figure(figsize=(5, 2.5))
    plt.loglog(greek_errors.steps, greek_errors.delta_errors, marker="o", label="Delta")
    plt.loglog(greek_errors.steps, greek_errors.gamma_errors, marker="s", label="Gamma")
    plt.loglog(greek_errors.steps, greek_errors.vega_errors, marker="^", label="Vega")
    plt.legend()
    plt.close()
    print("derivatives-hedging=passed " + " ".join(f"{key}={value:.6g}" for key, value in observed.items()))
    return 0


# %% [markdown]
# **中间证据与敏感性。** Delta/Gamma/Vega 的解析值与缩小步长的中心差分对照；
# 12/24/52 次调仓在同一规则下分别报告误差与成本，而不是挑选最漂亮的频率。
#
# **限制。** GBM、固定波动率和比例成本不含跳跃、波动率微笑、冲击或融资约束；
# 固定 seed 只给可复现回归，研究结论还需跨 seed 或 bootstrap 区间。

# %%
if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/derivatives-hedging/oracle.json")))
