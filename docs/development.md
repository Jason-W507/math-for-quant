# 开发、构建与发布

本页面向维护者。读者只需要 README 中的成品链接和阅读路线。

## 本地检查

```powershell
uv sync
uv run python tools/render_shared_registries.py --check
uv run python -m unittest discover -s tests
```

## 构建出版物

普通书稿构建使用 `tools/build_books.py`；上册、下册或共享答案册可以分别构建，正式发布时再构建全部产物。图形使用独立图源和缓存矢量资产，只有图源或冻结输入变化时才运行 `tools/build_figures.py`。

```powershell
uv run python tools/build_books.py --volume upper
uv run python tools/build_books.py --volume lower
uv run python tools/build_books.py --volume all
```

发布入口 `tools/build_release.py` 会检查工作树、课程清单、答案册、图形和 PDF 元数据，并生成带来源、版本、许可证和校验值的发行清单。路线级验证器位于 `tools/validate_*_route.py`。

## 编辑版文字检查

`tools/lint_prose.py` 只报告警告，不阻断构建。它检查工程术语在正文中的密度、`\MFQLead` 使用量、正文中的路径和重复标题；最终判断仍由人工通读完成。

```powershell
uv run python tools/lint_prose.py
```

更完整的材料分层约定见 [`evidence-contract.md`](evidence-contract.md)。
