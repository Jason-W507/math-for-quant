# %% [markdown]
# # 配对仿真与真实成交快照
#
# **研究目标。** 使用共同随机数比较静态与库存控制规则，并让真实 Coinbase 成交快照
# 只承担数据语义和数量级检查，不承担策略正确性的 oracle。

from pathlib import Path
import sys

from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture
from math_for_quant.lower.microstructure_route import run_simulation


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    result = run_simulation(fixture)
    assert_expected(result, oracle)
    assert result["simulation_events"] == 500
    assert result["real_trades"] == 20
    assert result["real_first_trade_id"] == 1064791075
    print("microstructure-simulation=passed")
    return 0


# %% [markdown]
# **故障注入与限制。** 若两种策略的订单方向哈希不同，比较立即失效。真实快照没有
# 深度、撤单与延迟；仿真输出只说明给定生成机制下的差异，不证明生产可交易性。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/microstructure-simulation/oracle.json")))
