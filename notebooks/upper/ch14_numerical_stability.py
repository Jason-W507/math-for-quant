# %% [markdown]
# # 浮点误差、稳定算法与病态线性系统
#
# Decimal 与解析公式提供独立 oracle；NumPy float64 是被测实现。

# %%
import runpy


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch14.numerical_stability", run_name="__main__")
