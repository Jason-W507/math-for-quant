# %% [markdown]
# # 凸优化、KKT 与对偶
#
# 手算 KKT、对偶、灵敏度和最小方差组合是独立 oracle；投影梯度只负责
# 数值交叉验证。非凸驻点和约束资格失败问题用于展示 KKT 边界。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch13.convex_optimization", run_name="__main__")
