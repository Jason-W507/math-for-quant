# %% [markdown]
# # 概率空间、分布变换与重尾反例
#
# 所有目标值先由有限求和或解析积分给出；程序只负责复算和交叉核验。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch07.probability_distributions", run_name="__main__")
