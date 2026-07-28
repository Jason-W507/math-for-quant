# 贡献指南

## 报告数学错误

请给出章节、公式或段落、所缺条件、反例或权威出处。不要只提交“看起来不对”的结论。

## 修改学习单元

1. 先更新权威 TeX 或 Jupytext 文本源；不要手改生成的 notebook、符号表、术语表或先修文件。
2. 数学、独立 oracle、四级题目与答案必须形成同一学习单元证据包。
3. 运行 `uv run python tools/render_shared_registries.py --check` 和相关单元测试。
4. 只构建受影响的册；上册改动使用 `--volume upper`，下册改动使用 `--volume lower`。
5. PDF 页数仅作异常诊断，不是内容完成或压缩目标。

完整验证命令见 README。
