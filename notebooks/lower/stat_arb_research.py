# %% [markdown]
# # 样本外交易：预测必须进入仓位和收益
#
# **研究目标。** 固定阈值、方向、仓位上限、持有期、调仓、部分成交、未成交与成本。
# **手算 oracle。** 目标仓位 `[1,-1,0,1]`，实际仓位 `[1,-0.5,0,0]`。

# %%
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture
from math_for_quant.lower.stat_arb_research import ExecutionPolicy, build_forecast_ledger, validate_purged_walk_forward


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
    validate_purged_walk_forward(
        train_indices=range(min(fixture["train_indices"]), max(fixture["train_indices"]) + 1),
        validation_indices=range(min(fixture["validation_indices"]), max(fixture["validation_indices"]) + 1),
        trade_indices=range(min(fixture["trade_indices"]), max(fixture["trade_indices"]) + 1),
        label_horizon=int(fixture["label_horizon"]), embargo=int(fixture["embargo"]),
    )
    plt.figure(figsize=(5, 2.5)); plt.step(range(4), ledger.target_positions, label="target"); plt.step(range(4), ledger.filled_positions, label="filled"); plt.legend(); plt.close()
    rejected = 0
    try:
        validate_purged_walk_forward(train_indices=range(0, 7), validation_indices=range(8, 10), trade_indices=range(12, 14), label_horizon=2, embargo=2)
    except ValueError:
        rejected = 1
    observed = {
        "gross_return": ledger.gross_return,
        "turnover": ledger.turnover,
        "cost": ledger.cost,
        "net_return": ledger.net_return,
        "unfilled_orders": int(np.sum((ledger.target_positions != 0.0) & (ledger.fill_fractions == 0.0))),
        "purge_failure_rejected": rejected,
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
