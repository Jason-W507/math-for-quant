# %% [markdown]
# # 研究有效性、样本外与交易摩擦
#
# 时间不等式、多重检验阈值和 gross-cost-net 手算账本提供独立 oracle。

# %%
import runpy
from pathlib import Path
import sys

ROOT = (Path(__file__).resolve().parents[2] if '__file__' in globals() else Path.cwd())
sys.path.insert(0, str(ROOT / 'src'))


if __name__ == "__main__":
    runpy.run_module("math_for_quant.ch16.research_validity", run_name="__main__")
