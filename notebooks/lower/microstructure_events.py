# %% [markdown]
# # 事件流、条件强度与队列成交
#
# **研究目标。** 从等待时间、订单方向和 FIFO 状态出发，区分 Poisson
# 基线、Hawkes 自激、季节性以及订单流—价格联合模型；再把队列位置转成成交概率。
#
# **显式假设与单位。** 时间单位为秒；Hawkes 观察窗右端为 2.5 秒，分支比必须
# 小于 1；SEC cancel-to-trade 比率无量纲，只有与 fixture 声明的每秒终止事件率
# 相乘后才成为队列耗尽强度。聚合比率不等于逐单队列位置。
# **手算 oracle。** 等待时间总和为 4、事件数为 4，所以 MLE 强度为 1；
# 方向 `[-1,-1,1,1]` 与价格变化 `[-.2,-.1,.1,.2]` 的过原点斜率为 .15。
# 显式观察窗 `[0,2.5]` 上的 Hawkes 对数似然由递推和数值积分独立重建；
# 季节调整等待时间的变换后残差均值必须为 1。

from pathlib import Path
import sys
import numpy as np

from math_for_quant.lower.notebook_evidence import (
    assert_expected, expect_value_error, load_oracle_and_fixture,
)
from math_for_quant.lower.microstructure_events import hawkes_log_likelihood
from math_for_quant.lower.microstructure_simulation import analyze_sec_order_placement
from math_for_quant.lower.microstructure_route import run_events


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    result = run_events(fixture)
    assert_expected(result, oracle)
    assert result["poisson_intensity"] == 1.0
    assert abs(float(result["joint_beta"]) - 0.15) < 1e-12
    assert result["queue_ahead"] == 3
    assert abs(float(result["hawkes_log_likelihood"]) + 2.9403405029867584) < 1e-12
    assert abs(float(result["seasonal_residual_mean"]) - 1.0) < 1e-12
    print("| model | estimate | diagnostic |")
    print("|---|---:|---:|")
    print(f"| Poisson | {result['poisson_intensity']:.6f} | {result['poisson_log_likelihood']:.6f} |")
    print(f"| seasonal Poisson | {result['seasonal_intensity']:.6f} | {result['seasonal_residual_mean']:.6f} |")
    print(f"| Hawkes | {result['hawkes_branching_ratio']:.6f} | {result['hawkes_log_likelihood']:.6f} |")

    sensitivity = []
    for rate in (5.0, 10.0, 20.0):
        changed = dict(fixture, real_depletion_event_rate=rate)
        sensitivity.append((rate, run_events(changed)["real_queue_fill_probability"]))
    print("sensitivity terminal_events_per_second -> full_fill_probability")
    for rate, probability in sensitivity:
        print(f"{rate:.1f} -> {probability:.6f}")

    failures = 0
    failures += expect_value_error(
        lambda: hawkes_log_likelihood(
            np.asarray(fixture["hawkes_times"], dtype=float),
            float(fixture["hawkes_baseline"]), float(fixture["hawkes_alpha"]),
            float(fixture["hawkes_beta"]), horizon=1.9,
        ),
        "horizon",
    )
    failures += expect_value_error(
        lambda: analyze_sec_order_placement(
            Path("tests/fixtures/sec-order-placement-invalid.json")
        ),
        "declared domains",
    )
    assert failures == 2
    print("microstructure-events=passed")
    return 0


# %% [markdown]
# **故障注入与限制。** 缩短观察窗使最后事件落在窗外、令分支比不小于 1，或给
# SEC 摘要传入非有限比率时都必须拒绝；上面的两个负例会在 notebook 执行时真实
# 运行。聚合 cancel-to-trade 比率不是逐单队列位置；敏感性表把额外的事件率假设
# 单独扫描，避免把无量纲概率偷偷当作每秒强度。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/microstructure-events/oracle.json")))
