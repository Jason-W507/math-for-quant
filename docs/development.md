# 开发、构建与发布

本页面向维护者。读者只需要 README 中的成品链接和阅读路线。

## 本地检查

```powershell
uv sync
$env:JUPYTER_ALLOW_INSECURE_WRITES = "true"
$env:IPYTHONDIR = "build/ipython"
$env:JUPYTER_RUNTIME_DIR = "build/jupyter-runtime"
New-Item -ItemType Directory -Force build/ipython, build/jupyter-runtime | Out-Null
uv run jupyter notebook notebooks/foundation/independent_oracle.ipynb
# 无界面时检查单个 notebook：
uv run jupyter nbconvert --to notebook --execute --stdout `
  --ExecutePreprocessor.timeout=60 `
  --ExecutePreprocessor.allow_error_names=SystemExit `
  notebooks/foundation/independent_oracle.ipynb
```

## 构建出版物

普通书稿构建使用 `tools/publish.ps1`；上册或下册可以分别构建，正式发布时加 `-Release`。图形使用独立图源和缓存矢量资产，只有图源或冻结输入变化时才运行 `tools/build_figures.py`。

```powershell
pwsh tools/publish.ps1 -Volume upper
pwsh tools/publish.ps1 -Volume lower
pwsh tools/publish.ps1 -Volume all
```

发布入口 `tools/publish.ps1 -Release` 会打包仓库中跟踪的 `.ipynb`、构建三份 PDF，并直接生成 notebook 压缩包；它不会为了发布无条件重新执行整套课程。需要检查 notebook 时，只运行本次修改的文件。`latexmk` 负责报告 LaTeX 编译错误；版式通过打开或截图 `output/pdf/` 中的 PDF 检查。

## 编辑版文字检查

`tools/lint_prose.py` 只报告警告，不阻断构建。它检查工程术语在正文中的密度、`\MFQLead` 使用量、正文中的路径和重复标题；最终判断仍由人工通读完成。

```powershell
uv run python tools/lint_prose.py
```

更完整的材料分层约定见 [`evidence-contract.md`](evidence-contract.md)。
