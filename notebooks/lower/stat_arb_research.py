# %% [markdown]
# # 样本外交易：预测必须进入仓位和收益
#
# **研究目标。** 固定阈值、方向、仓位上限、持有期、调仓、部分成交、未成交与成本。
# **假设。** 决策在收益实现前完成；成交比例作用于订单增量；成本按实际仓位变化计提。
# **手算 oracle。** 目标仓位 `[1,-1,0,1]`。第二期从 `+1` 到 `-1` 的订单为 `-2`，成交一半后实际仓位为 `0`；最后一期零成交保持上一期仓位。
# **敏感性。** 将单位换手成本加倍，净收益下降量必须等于新增成本。

# %%
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture
from math_for_quant.lower.stat_arb import validate_scaler, validate_walk_forward
from math_for_quant.lower.stat_arb_research import ExecutionPolicy, build_forecast_ledger, validate_failure_state, validate_purged_walk_forward
from math_for_quant.lower.stat_arb_execution_library import library_forecast_ledger


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    policy = ExecutionPolicy(
        entry_threshold=float(fixture["entry_threshold"]),
        position_limit=float(fixture["position_limit"]),
        holding_period=int(fixture["holding_period"]),
        rebalance_every=int(fixture["rebalance_every"]),
        cost_per_unit_turnover=float(fixture["cost_per_unit_turnover"]),
    )
    ledger = build_forecast_ledger(
        forecasts=np.asarray(fixture["forecasts"], dtype=float),
        realized_returns=np.asarray(fixture["realized_returns"], dtype=float),
        fill_fractions=np.asarray(fixture["fill_fractions"], dtype=float),
        policy=policy,
    )
    library_ledger = library_forecast_ledger(
        forecasts=np.asarray(fixture["forecasts"], dtype=float),
        realized_returns=np.asarray(fixture["realized_returns"], dtype=float),
        fill_fractions=np.asarray(fixture["fill_fractions"], dtype=float),
        policy=policy,
    )
    validate_purged_walk_forward(
        train_indices=range(min(fixture["train_indices"]), max(fixture["train_indices"]) + 1),
        validation_indices=range(min(fixture["validation_indices"]), max(fixture["validation_indices"]) + 1),
        trade_indices=range(min(fixture["trade_indices"]), max(fixture["trade_indices"]) + 1),
        label_horizon=int(fixture["label_horizon"]), embargo=int(fixture["embargo"]),
    )
    plt.figure(figsize=(5, 2.5)); plt.step(range(4), ledger.target_positions, label="target"); plt.step(range(4), ledger.filled_positions, label="filled"); plt.legend(); plt.close()
    rejected = {"purge": 0, "embargo": 0, "walk_forward": 0, "scaler": 0, "failure": 0}
    try:
        validate_purged_walk_forward(train_indices=range(0, 7), validation_indices=range(8, 10), trade_indices=range(12, 14), label_horizon=2, embargo=2)
    except ValueError:
        rejected["purge"] = 1
    try:
        validate_purged_walk_forward(train_indices=range(0, 6), validation_indices=range(8, 10), trade_indices=range(11, 14), label_horizon=2, embargo=2)
    except ValueError:
        rejected["embargo"] = 1
    try:
        validate_walk_forward([
            {"name":"train", "start":"2020-01", "end":"2020-06"},
            {"name":"validation", "start":"2020-06", "end":"2020-08"},
            {"name":"trade", "start":"2020-09", "end":"2020-12"},
        ])
    except ValueError:
        rejected["walk_forward"] = 1
    try:
        validate_scaler("2020-08-01", "2020-06-01")
    except ValueError:
        rejected["scaler"] = 1
    try:
        validate_failure_state(half_life=9.0, maximum_half_life=4.0, fill_rate=0.9, minimum_fill_rate=0.5)
    except ValueError:
        rejected["failure"] = 1
    high_cost = build_forecast_ledger(
        forecasts=np.asarray(fixture["forecasts"], dtype=float),
        realized_returns=np.asarray(fixture["realized_returns"], dtype=float),
        fill_fractions=np.asarray(fixture["fill_fractions"], dtype=float),
        policy=ExecutionPolicy(policy.entry_threshold, policy.position_limit, policy.holding_period, policy.rebalance_every, 2.0 * policy.cost_per_unit_turnover),
    )
    observed = {
        "gross_return": ledger.gross_return,
        "turnover": ledger.turnover,
        "cost": ledger.cost,
        "net_return": ledger.net_return,
        "unfilled_orders": int(np.sum((ledger.target_positions != 0.0) & (ledger.fill_fractions == 0.0))),
        "purge_failure_rejected": rejected["purge"],
        "embargo_failure_rejected": rejected["embargo"],
        "walk_forward_failure_rejected": rejected["walk_forward"],
        "scaler_failure_rejected": rejected["scaler"],
        "failure_state_rejected": rejected["failure"],
        "cost_sensitivity": ledger.net_return - high_cost.net_return,
        "execution_library_max_gap": max(
            float(np.max(np.abs(ledger.filled_positions - library_ledger.filled_positions))),
            abs(ledger.net_return - library_ledger.net_return),
        ),
    }
    assert_expected(observed, oracle)
    print("stat-arb-research=passed " + " ".join(f"{key}={value:.6f}" for key, value in observed.items()))
    return 0


# %% [markdown]
# **限制。** 合成成交比例只验证账本；真实队列、借券、冲击与交易暂停必须另建执行模型。

# %%
if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/stat-arb-research/oracle.json")
    raise SystemExit(main(path))
