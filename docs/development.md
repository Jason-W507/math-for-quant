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

课程注册表、先修地图和来源映射题覆盖表由一个入口生成；只有相应的 `curriculum/` JSON 修改时才需要运行：

```powershell
uv run python tools/build_course_assets.py
```

其中 `curriculum/interview-problem-ledger.json` 是来源映射题台账，脚本只把它渲染成书内覆盖表，不再另有一份 Python 题库生成器。

## 编辑版文字检查

`tools/check.py prose` 只报告警告，不阻断构建。它检查工程术语在正文中的密度、`\MFQLead` 使用量、正文中的路径和重复标题；最终判断仍由人工通读完成。模板来源检查使用同一个入口的 `template` 子命令。

```powershell
uv run python tools/check.py prose
uv run python tools/check.py template --vendored-only
```

下册路线报告共用一个入口；只在对应路线的代码或冻结输入变化时运行它：

```powershell
uv run python tools/build_reports.py multifactor
uv run python tools/build_reports.py stat-arb
uv run python tools/build_reports.py ml-alpha
uv run python tools/build_reports.py derivatives
uv run python tools/build_reports.py portfolio-risk
uv run python tools/build_reports.py microstructure
```

更完整的材料分层约定见 [`evidence-contract.md`](evidence-contract.md)。
