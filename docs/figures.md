# 全书图形系统

## 角色

- 概念图解释结构、依赖、几何关系或信息流，不充当经验数据。
- 证据图由冻结参数和确定性计算生成，用于展示数值形状、误差、分布或敏感性。
- 蓝色表示对象、模型和信息流；绿色表示有效条件、通过验证和可行区域；橙红表示泄漏、失败、风险和约束；灰色表示不可观测、冻结或背景。
- 颜色不是唯一编码。风险同时使用虚线或显式标签，有效路径和对象也通过形状、线型和文字区分。

## 图源与缓存

`figures/figure-specs.json` 是图形规格来源。`tools/build_figures.py` 读取规格并生成：

- `tex/figures/generated/*.pdf`：供 LaTeX 直接引用的缓存矢量资产；
- `tex/figures/generated/*.tex`：包含图注和图源 ID 的包装片段；
- `figures/figure-manifest.json`：记录图源、缓存路径与 SHA-256 的溯源清单。

每幅证据图还登记 `scene` 与 `companion`。`companion` 指向对应章节的可修改 teaching notebook；它提供参数实验入口，PDF 图本身仍保持可独立阅读。

显式更新图形：

```powershell
uv run python tools/build_figures.py
```

普通 `latexmk` 或 `tools/publish.ps1` 只读取缓存资产，不运行绘图脚本或数值实验。

## 章节嵌入

`tools/integrate_figures.py` 按规格中的章节锚点，将包装片段插入该节开头论述之后。命令是幂等的：已经嵌入的图不会重复插入。

新增图时应确保：

1. 正文在图前提出需要图回答的问题；
2. 图注陈述图能够支持的关系，并保留图源 ID；
3. 图后正文解释读法、适用条件和不可推出的结论；
4. 证据图不使用未登记输入，不把示意数据表述成经验结论；
5. 灰度打印时仍可通过线型、形状和标签读懂。

## 局部视觉检查

生成图形联系表而不重建全书：

```powershell
uv run python tools/render_figure_contact_sheets.py
```

联系表写入 `tmp/figures/`，只用于本地 QA，不属于出版资产。
