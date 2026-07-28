# %% [markdown]
# # Monte Carlo、bootstrap 与方差缩减
#
# 解析积分、解析方差和可枚举小样本提供独立 oracle；伪随机模拟只是被测实现。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch15.monte_carlo", run_name="__main__")
