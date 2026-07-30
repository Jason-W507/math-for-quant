# %% [markdown]
# # Alpha 决策与研究包：损失、仓位、成交、成本和文本时间
#
# **研究目标。** 把横截面分数映射为多空目标、部分成交、换手与净收益，并审计文本发布时间/修订时间。
# **假设。** 每期分数先于下一期收益；多空各一只且毛敞口为 2；成交比例作用于订单增量。
# **手算 oracle。** 第一期仓位 `[1,0,-1]` 毛收益 `0.03`；第二期实际仓位 `[0,1,-1]`
# 毛收益 `0.02`，总换手 4、成本 `0.004`、净收益 `0.046`。
# **失败注入。** 决策日后修订的文本必须被拒绝，不能回填到历史特征。

# %%
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.ml_alpha_research import AlphaLedgerInputs, PortfolioPolicy, build_alpha_ledger, cross_sectional_mse, pairwise_ranking_loss, return_weighted_loss
from math_for_quant.lower.ml_alpha_text import audit_text_timestamps
from math_for_quant.lower.ml_alpha_execution_library import library_alpha_ledger
from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    scores=np.asarray(fixture["scores"]); returns=np.asarray(fixture["realized_returns"]); fills=np.asarray(fixture["fill_fractions"])
    inputs=AlphaLedgerInputs(scores, returns, fills, PortfolioPolicy(int(fixture["long_count"]),int(fixture["short_count"]),float(fixture["gross_limit"]),float(fixture["cost_per_unit_turnover"])))
    ledger=build_alpha_ledger(inputs=inputs)
    library_ledger=library_alpha_ledger(inputs=inputs)
    audit_text_timestamps(publication_dates=fixture["publication_dates"], revision_dates=fixture["revision_dates"], decision_date=fixture["decision_date"])
    revision_rejected=0
    try: audit_text_timestamps(publication_dates=["2024-01-01"], revision_dates=["2024-01-05"], decision_date="2024-01-03")
    except ValueError: revision_rejected=1
    observed={"cross_sectional_mse":cross_sectional_mse(scores[0],returns[0]), "ranking_loss":pairwise_ranking_loss(scores[0],returns[0]), "return_weighted_loss":return_weighted_loss(scores[0],returns[0]), "gross_return":ledger.gross_return, "turnover":ledger.turnover, "cost":ledger.cost, "net_return":ledger.net_return, "execution_library_gap":max(float(np.max(np.abs(ledger.filled_positions-library_ledger.filled_positions))),abs(ledger.net_return-library_ledger.net_return)), "revision_leakage_rejected":revision_rejected, "unfilled_orders":int(np.sum(fills==0.0))}
    plt.figure(figsize=(5,2.5)); plt.plot(ledger.period_gross_returns, marker="o"); plt.close()
    assert_expected(observed,oracle)
    print("ml-alpha-research=passed "+" ".join(f"{key}={value:.6f}" for key,value in observed.items()))
    return 0


# %% [markdown]
# **敏感性。** 报告 MSE、排序与收益加权三种目标；改变目标会改变模型选择与仓位，不应只比较一个分数。
# **限制。** 冻结账本没有冲击、借券和容量模型；开放 Capstone 必须补足并报告失败边界。

# %%
if __name__=="__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv)>1 else Path("evidence/ml-alpha-research/oracle.json")))
