# %% [markdown]
# # 配对仿真与 SEC 订单位置压力轨
#
# **研究目标。** 使用共同随机源比较静态与库存反馈报价；SEC 公共领域
# 订单位置摘要提供压力成交概率，但不承担策略正确性的 oracle。
#
# **显式假设与单位。** 两种策略共享到达间隔、方向、价格创新和成交均匀数，但报价
# 各自决定成交阈值；现金与库存都以一手合约记账，终点用同一中价盯市。

# %%
from pathlib import Path
import sys

from math_for_quant.lower.notebook_evidence import (
    assert_expected, expect_value_error, load_oracle_and_fixture,
)
from math_for_quant.lower.microstructure_route import run_simulation
from math_for_quant.lower.microstructure_simulation import paired_event_simulation


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    result = run_simulation(fixture)
    assert_expected(result, oracle)
    assert result["simulation_events"] == 500
    assert result["baseline_fills"] == 21
    assert result["control_fills"] == 23
    assert result["real_categories"] == 5
    assert abs(float(result["real_weighted_cancel_to_trade"]) - 55.3908) < 1e-10
    print("| policy | fills | terminal inventory | max abs inventory | pnl |")
    print("|---|---:|---:|---:|---:|")
    print(f"| static | {result['baseline_fills']} | {result['baseline_ending_inventory']} | {result['baseline_max_abs_inventory']} | {result['baseline_pnl']:.6f} |")
    print(f"| inventory feedback | {result['control_fills']} | {result['control_ending_inventory']} | {result['control_max_abs_inventory']} | {result['control_pnl']:.6f} |")

    base_probability = float(result["real_implied_execution_probability"])
    print("sensitivity fill_probability -> static/control fills")
    for probability in (base_probability / 2.0, base_probability, base_probability * 2.0):
        observed = paired_event_simulation(
            seed=int(fixture["seed"]), events=int(fixture["events"]),
            base_intensity=float(fixture["base_intensity"]), fill_probability=probability,
        )
        print(f"{probability:.6f} -> {observed.baseline_fills}/{observed.control_fills}")
    failure = expect_value_error(
        lambda: paired_event_simulation(
            seed=1, events=10, base_intensity=1.0, fill_probability=1.2
        ),
        "invalid",
    )
    assert failure == 1
    print("microstructure-simulation=passed")
    return 0


# %% [markdown]
# **故障注入与限制。** 若两种策略的订单方向或成交掩码哈希不同，比较立即失效。
# 每笔成交用“现金 + 期末库存按终点中价盯市”记账，主动买令做市商库存减少。SEC
# 摘要没有逐档深度、订单方向与延迟；仿真输出只说明给定机制下的差异。上面的
# 非法概率负例和三档敏感性扫描会随 notebook 一起执行。

# %%
if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/microstructure-simulation/oracle.json")))
