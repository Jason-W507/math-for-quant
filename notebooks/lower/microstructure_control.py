# %% [markdown]
# # 执行控制与库存反馈
#
# **研究目标。** 把计划数量、随机完成率、临时/永久冲击和停止规则放进同一执行账本；
# 再验证一次成交如何改变库存，并反馈到下一轮双边报价。
#
# **显式假设与单位。** 数量为手、价格为每手现金价格；临时冲击乘本期成交量，
# 永久冲击乘此前累计成交量。期限耗尽且仍有剩余时，`stopped` 必须为真。

from pathlib import Path
import sys

from math_for_quant.lower.notebook_evidence import (
    assert_expected, expect_value_error, load_oracle_and_fixture,
)
from math_for_quant.lower.microstructure_control import execution_path
from math_for_quant.lower.microstructure_route import run_control


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    result = run_control(fixture)
    assert_expected(result, oracle)
    assert result["execution_filled"] == 5
    assert result["execution_remaining"] == 4
    assert result["maker_inventory"] == 1
    assert result["maker_next_bid"] < 99.5
    assert result["real_execution_remaining"] == 8
    print("| path | filled | remaining | shortfall |")
    print("|---|---:|---:|---:|")
    print(f"| synthetic | {result['execution_filled']} | {result['execution_remaining']} | {result['execution_shortfall']:.6f} |")
    print(f"| SEC stress | {result['real_execution_filled']} | {result['real_execution_remaining']} | {result['real_execution_shortfall']:.6f} |")

    print("sensitivity temporary_impact -> shortfall")
    for impact in (0.1, 0.2, 0.4):
        changed = dict(fixture, temporary_impact=impact)
        print(f"{impact:.1f} -> {run_control(changed)['execution_shortfall']:.6f}")
    failure = expect_value_error(
        lambda: execution_path(
            schedule=[1], fill_rates=[1.2], initial_inventory=1,
            arrival_price=100.0, temporary_impact=0.2,
            permanent_impact=0.1, stop_after=1,
        ),
        "fill rate",
    )
    assert failure == 1
    print("microstructure-control=passed")
    return 0


# %% [markdown]
# **敏感性与限制。** 逐一扫描冲击系数、库存惩罚和最大切片；停止不是“把剩余数量
# 当作成交”，而是保留未完成状态。SEC 各位置隐含执行概率会真正生成一条压力路径，
# 但聚合比率没有母单、到达价和逐笔轨迹，不能估计现实 IS。这个教学 DP 没有延迟、
# 盘口恢复和机会成本估计。上面的非法完成率负例与冲击敏感性表均在 notebook
# 执行时生成，而非只写在 Markdown 中。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/microstructure-control/oracle.json")))
