# %% [markdown]
# # 多因子研究：从信号到仓位、换手与净收益
#
# **研究目标。** 显式实现信号、分组、持有期、权重、成交摩擦和净收益链。
# **假设。** 信号在调仓前可得；收益在之后实现；合成 fixture 是正确性 oracle。
# **手算 oracle。** 两期等权多空毛收益为 3.5% 与 3.0%，均值 3.25%。

# %%
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.multifactor_research import (
    PortfolioInputs,
    PortfolioPolicy,
    Weighting,
    build_group_portfolio_ledger,
    load_real_cross_section,
)
from math_for_quant.lower.multifactor import validate_time_boundary
from math_for_quant.lower.notebook_evidence import (
    assert_expected,
    load_oracle_and_fixture,
)


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    for signal_date, return_date in zip(
        fixture["signal_dates"], fixture["return_dates"], strict=True
    ):
        validate_time_boundary(signal_date, return_date)
    inputs = PortfolioInputs(
        signals=np.asarray(fixture["signals"], dtype=float),
        realized_returns=np.asarray(fixture["realized_returns"], dtype=float),
        market_caps=np.asarray(fixture["market_caps"], dtype=float),
    )
    ledger = build_group_portfolio_ledger(
        inputs=inputs,
        policy=PortfolioPolicy(
        quantiles=int(fixture["quantiles"]),
        weighting=Weighting.EQUAL,
        holding_periods=int(fixture["holding_periods"]),
        cost_per_unit_turnover=float(fixture["cost_per_unit_turnover"]),
        capacity_impact=float(fixture["capacity_impact"]),
        ),
    )
    capitalized = build_group_portfolio_ledger(
        inputs=inputs,
        policy=PortfolioPolicy(
        quantiles=int(fixture["quantiles"]),
        weighting=Weighting.CAPITALIZATION,
        holding_periods=2,
        cost_per_unit_turnover=float(fixture["cost_per_unit_turnover"]),
        capacity_impact=float(fixture["capacity_impact"]),
        ),
    )
    real = load_real_cross_section(Path("data/real/multifactor-wdi-2013-2014.json"))

    # 图表：毛收益、成本和容量冲击必须并列显示。
    plt.figure(figsize=(5, 2.5))
    plt.bar(["gross", "cost", "impact", "net"], [ledger.gross_return, ledger.cost, ledger.capacity_impact, ledger.net_return])
    plt.close()

    # 故障注入：信号与收益面板错位必须拒绝。
    misalignment_rejected = 0
    try:
        build_group_portfolio_ledger(
            inputs=PortfolioInputs(
                signals=inputs.signals,
                realized_returns=inputs.realized_returns[:, :-1],
                market_caps=inputs.market_caps,
            ),
            policy=PortfolioPolicy(
                quantiles=2,
                weighting=Weighting.EQUAL,
                holding_periods=1,
                cost_per_unit_turnover=0.0,
                capacity_impact=0.0,
            ),
        )
    except ValueError:
        misalignment_rejected = 1
    observed = {
        "gross": ledger.gross_return,
        "net": ledger.net_return,
        "turnover": ledger.turnover,
        "capitalization_net": capitalized.net_return,
        "real_correlation": real.correlation,
        "misalignment_rejected": misalignment_rejected,
    }
    assert_expected(observed, oracle)
    print(
        f"multifactor-research=passed gross={ledger.gross_return:.6f} net={ledger.net_return:.6f} "
        f"turnover={ledger.turnover:.6f} cap_weight_net={capitalized.net_return:.6f} "
        f"wdi_correlation={real.correlation:.6f} misalignment_rejected={misalignment_rejected}"
    )
    return 0


# %% [markdown]
# **限制。** WDI 截面只展示外部数据、许可、哈希和时间协议；它不是股票 alpha 证据。

# %%
if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/multifactor-research/oracle.json")
    raise SystemExit(main(path))
