# %% [markdown]
# # Capstone：可复现研究审计
#
# 独立 oracle 固定数据校验值、时间边界、多重检验、成本账本、数值探针、许可与限制；审计程序只负责拒绝不满足声明的研究包。

# %%
import runpy


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch17.research_audit", run_name="__main__")
