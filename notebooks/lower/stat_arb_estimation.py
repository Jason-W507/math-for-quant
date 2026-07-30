# %% [markdown]
# # 估计与状态推断：滤波不等于平滑
#
# **研究目标。** 比较在线 Kalman 滤波、全样本平滑、两状态 Bayes 过滤与已知变点。
# **失败注入。** 决策时使用平滑状态必须被时间边界拒绝。

# %%
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture
from math_for_quant.lower.stat_arb import detect_alarms, validate_online_state
from math_for_quant.lower.stat_arb_estimation import kalman_filter_and_smooth, regime_filter


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    observations = np.asarray(fixture["observations"], dtype=float)
    states = kalman_filter_and_smooth(
        observations,
        transition=float(fixture["transition"]),
        observation_loading=float(fixture["observation_loading"]),
        process_variance=float(fixture["process_variance"]),
        observation_variance=float(fixture["observation_variance"]),
        initial_mean=float(fixture["initial_mean"]),
        initial_variance=float(fixture["initial_variance"]),
    )
    regimes = regime_filter(
        observations,
        transition=np.asarray(fixture["regime_transition"], dtype=float),
        means=np.asarray(fixture["regime_means"], dtype=float),
        variances=np.asarray(fixture["regime_variances"], dtype=float),
        initial=np.asarray(fixture["regime_initial"], dtype=float),
    )
    plt.figure(figsize=(5, 2.5)); plt.plot(states.filtered, label="filtered"); plt.plot(states.smoothed, label="smoothed"); plt.legend(); plt.close()
    rejected = 0
    try:
        validate_online_state("2020-12-31", "2020-06-30")
    except ValueError:
        rejected = 1
    alarms = detect_alarms(observations, minimum_segment=2, threshold=2.5)
    observed = {
        "filtered_last": states.filtered[-1],
        "smoothed_second": states.smoothed[1],
        "regime_one_last": regimes[-1, 1],
        "future_smoother_rejected": rejected,
        "change_index": alarms[0],
    }
    assert_expected(observed, oracle)
    print("stat-arb-estimation=passed " + " ".join(f"{key}={value:.6f}" for key, value in observed.items()))
    return 0


# %% [markdown]
# **限制。** 两状态高斯模型只验证概率递推；状态名称、分布选择与结构突变仍有识别风险。

# %%
if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/stat-arb-estimation/oracle.json")
    raise SystemExit(main(path))
