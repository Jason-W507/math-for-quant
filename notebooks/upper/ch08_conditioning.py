# %% [markdown]
# # 条件期望、嵌套信息、Bayes 与两类独立性反例
#
# 所有目标值来自有限状态手算账本；程序只复算定义积分、投影恒等式和概率分解。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch08.conditioning", run_name="__main__")
