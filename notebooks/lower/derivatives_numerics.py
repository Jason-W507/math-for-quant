# %% [markdown]
# # 定价与校准：闭式、树、PDE、Monte Carlo 与曲面
#
# **研究目标。** 对同一合约分开报告离散、截断和采样误差，并把逐点隐波反演与
# 参数化总方差曲面校准分成两步。
#
# **假设。** 欧式看涨、常利率/波动率、无跳跃；PDE 使用有限价格域，Monte Carlo
# 使用固定伪随机样本。财政部快照只提供贴现输入的 provenance。
#
# **手算 oracle。** Black--Scholes 闭式是共同基准；树和 PDE 应落在声明的离散误差
# 预算内，Monte Carlo 偏差应结合标准误解释。参数曲面的三个系数在生成报价前冻结。
#
# **失败注入。** 非等距执行价必须比较相邻斜率；含分红或负利率时，原始固定执行价
# 日历门禁必须拒绝，只有 forward/discount 归一化后的同 moneyness 网格才可比较。

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
    plt.bar(
        ["space", "time", "boundary", "sampling"],
        [
            observed["pde_space_gap"], observed["pde_time_gap"],
            observed["pde_boundary_gap"], observed["mc_standard_error"],
        ],
    )
    plt.close()
    if observed["forward_calendar_passed"] != 1:
        raise SystemExit("forward-normalized calendar experiment did not pass")
    if observed["raw_calendar_rejected"] != 1:
        raise SystemExit("raw calendar gate failed to reject dividend assumptions")
    if observed["nonuniform_convexity_rejected"] != 1:
        raise SystemExit("nonuniform convexity failure injection did not reject")
    if max(
        observed["closed_library_gap"], observed["tree_library_gap"],
        observed["pde_library_gap"], observed["mc_library_gap"],
    ) > 1e-10:
        raise SystemExit("transparent/mature-library pricing paths diverged")
    print("derivatives-numerics=passed " + " ".join(f"{key}={value:.6g}" for key, value in observed.items()))
    return 0


# %% [markdown]
# **中间证据与敏感性。** 图中分别保留空间网格、时间网格、价格域边界和采样标准误，
# 因而总 PDE 偏差不会被错误地贴成单一“离散误差”。透明 Thomas 求解与 SciPy
# `solve_banded`、逐层树与 SciPy 二项分布、手写正态 CDF 与 SciPy `ndtr` 分别对照。
#
# **限制。** 合成曲面由模型自身生成，只能验证恢复与约束；真实报价还需要 bid/ask、
# 报价时间、远期和贴现曲线。固定网格的误差差值是诊断，不是严格误差上界。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/derivatives-numerics/oracle.json")))
