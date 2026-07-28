# %% [markdown]
# # 估计、异方差标准误与多重检验
#
# 解析矩、固定设计真方差和独立检验族的精确 FWER 均由 oracle 给出；
# 固定种子模拟只检验声明的名义性质。

# %%
import runpy


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch10.statistical_inference", run_name="__main__")
