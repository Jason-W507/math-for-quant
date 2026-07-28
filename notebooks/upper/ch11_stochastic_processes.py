# %% [markdown]
# # Markov、鞅、Poisson 与 Brownian 过程
#
# 矩阵幂、稳态方程、Poisson 矩、Brownian 协方差与有限空间条件概率
# 均由解析 oracle 给出；模拟只做固定容差的交叉验证。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch11.stochastic_processes", run_name="__main__")
