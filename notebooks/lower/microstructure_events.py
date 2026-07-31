# %% [markdown]
# # 事件流、条件强度与队列成交
#
# **研究目标。** 从等待时间、订单方向和 FIFO 状态出发，区分 Poisson
# 基线、Hawkes 自激、季节性以及订单流—价格联合模型；再把队列位置转成成交概率。
# **手算 oracle。** 等待时间总和为 4、事件数为 4，所以 MLE 强度为 1；
# 方向 `[-1,-1,1,1]` 与价格变化 `[-.2,-.1,.1,.2]` 的过原点斜率为 .15。

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
    print("microstructure-events=passed")
    return 0


# %% [markdown]
# **故障注入与限制。** 把 trade id 打乱时真实数据解析必须拒绝；公开成交并不等于
# 可见订单簿，因此不能用它反推真实队列位置。扫描窗口长度与季节分桶，记录参数漂移。

if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/microstructure-events/oracle.json")))
