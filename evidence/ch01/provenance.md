# 第 1 章 fixture 与 oracle 来源

`data/fixtures/ch01.json` 是凸组合界和删除非负性假设反例的固定输入。期望值由书稿中的手算账本给出，不由被测程序生成；`oracle.json` 通过 fixture 文件的 SHA-256 绑定这组输入，使输入变化必须显式更新来源记录与独立答案。
