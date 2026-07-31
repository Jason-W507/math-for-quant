# %% [markdown]
# # 事件流、条件强度与队列成交
#
# **研究目标。** 从等待时间、订单方向和 FIFO 状态出发，区分 Poisson
# 基线、Hawkes 自激、季节性以及订单流—价格联合模型；再把队列位置转成成交概率。
# **手算 oracle。** 等待时间总和为 4、事件数为 4，所以 MLE 强度为 1；
# 方向 `[-1,-1,1,1]` 与价格变化 `[-.2,-.1,.1,.2]` 的过原点斜率为 .15。
# 显式观察窗 `[0,2.5]` 上的 Hawkes 对数似然由递推和数值积分独立重建；
# 季节调整等待时间的变换后残差均值必须为 1。

from pathlib import Path
import sys

from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture
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
    print("microstructure-events=passed")
    return 0


# %% [markdown]
# **故障注入与限制。** 缩短观察窗使最后事件落在窗外、令分支比不小于 1，或给
# SEC 摘要传入非有限比率时都必须拒绝。聚合 cancel-to-trade 比率不是逐单队列位置；
# 它只驱动明确标注的压力概率。扫描窗口长度与季节分桶时应记录参数漂移。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/microstructure-events/oracle.json")))
