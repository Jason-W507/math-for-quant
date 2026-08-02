# 贡献指南

## 报告数学错误

请给出章节、公式或段落、所缺条件、反例或权威出处。不要只提交“看起来不对”的结论。

## 修改学习单元

1. 先更新权威 TeX 或 `notebooks/` 下受 Git 跟踪的 `.ipynb`；不要手改 `output/` 中的发布打包副本、符号表、术语表或先修文件。
2. 数学、独立 oracle、四级题目与答案必须形成同一学习单元证据包。
3. 直接运行受影响的 notebook；需要出版物时运行 `pwsh tools/publish.ps1 -Volume upper|lower`，用 `latexmk` 输出和 PDF 截图检查结果。
4. 只构建受影响的册；上册改动使用 `-Volume upper`，下册改动使用 `-Volume lower`。
5. PDF 页数仅作异常诊断，不是内容完成或压缩目标。

完整验证命令见 README。
