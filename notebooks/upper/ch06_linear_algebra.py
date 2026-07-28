# %% [markdown]
# # 投影、最小二乘、SVD 与近奇异放大
#
# 正规方程、残差和扰动解先由手算给出；NumPy 负责独立分解与重构核验。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch06.linear_algebra", run_name="__main__")
