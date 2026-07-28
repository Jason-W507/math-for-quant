# %% [markdown]
# # Lp 范数、求积误差与 Fubini 反例
#
# 解析范数、复合中点余项和双重级数计数先由手算给出；程序只复现这些 oracle。

# %%
import runpy


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch04.lp_product_measure", run_name="__main__")
