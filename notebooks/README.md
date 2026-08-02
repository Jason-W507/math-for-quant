# 教学 notebooks

这里的 `.ipynb` 文件是本书可执行教材的权威来源，也是读者应当打开和修改的教学界面。它们受 Git 跟踪，不需要先经过转换步骤。

公共计算实现放在 `src/math_for_quant/`；notebook 负责提出问题、展示推导与运行实验。直接用 Jupyter 执行任意 notebook：

```powershell
$env:JUPYTER_ALLOW_INSECURE_WRITES = "true"
$env:IPYTHONDIR = "build/ipython"
$env:JUPYTER_RUNTIME_DIR = "build/jupyter-runtime"
New-Item -ItemType Directory -Force build/ipython, build/jupyter-runtime | Out-Null
uv run jupyter notebook notebooks/foundation/independent_oracle.ipynb
# 无界面或发布时使用：
uv run jupyter nbconvert --to notebook --execute --stdout `
  --ExecutePreprocessor.timeout=60 `
  --ExecutePreprocessor.allow_error_names=SystemExit `
  notebooks/foundation/independent_oracle.ipynb
```

正式发布时，`tools/publish.ps1 -Release` 会把执行后的副本写入被忽略的 `output/notebooks/`，并将它们打包。不要手工编辑 `output/` 中的文件；重新执行发布命令即可得到新的证据副本。
