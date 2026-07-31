# %% [markdown]
# # 配对仿真与 SEC 订单位置压力轨
#
# **研究目标。** 使用共同随机数和同一成交掩码比较静态与库存反馈报价；SEC 公共领域
# 订单位置摘要提供压力成交概率，但不承担策略正确性的 oracle。

from pathlib import Path
import sys

from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture
from math_for_quant.lower.microstructure_route import run_simulation


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    result = run_simulation(fixture)
    assert_expected(result, oracle)
    assert result["simulation_events"] == 500
    assert result["simulation_fills"] == 21
    assert result["real_categories"] == 5
    assert abs(float(result["real_weighted_cancel_to_trade"]) - 55.3908) < 1e-10
    print("microstructure-simulation=passed")
    return 0


# %% [markdown]
# **故障注入与限制。** 若两种策略的订单方向或成交掩码哈希不同，比较立即失效。
# 每笔成交用“现金 + 期末库存按终点中价盯市”记账，主动买令做市商库存减少。SEC
# 摘要没有逐档深度、订单方向与延迟；仿真输出只说明给定机制下的差异。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/microstructure-simulation/oracle.json")))
