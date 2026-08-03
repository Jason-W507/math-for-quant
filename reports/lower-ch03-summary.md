# 机器学习 Alpha 可复现研究包

- 数据生成过程：`y = 0.2 * x + 0.8 * 1[x > 0]`，目标由代码生成而非预填预测。
- 训练窗口：索引 0--4；验证窗口：索引 5--6；固定推理窗口：索引 7--8。
- 切分审计：随机切分：拒绝；全样本预处理：拒绝；未来目标对齐：拒绝；张量维度错误：拒绝。
- 模型选择：验证集 MSE 为 baseline=1.0634、tree=0.0650、boosting=0.0425；冻结选择 boosting。
- 固定推理：MSE 为 baseline=1.5154、tree=0.2050、boosting=0.1625；模型指纹 `1ee0b9a0c428`。
- 序列任务：非深度基线 MSE=6.398979；冻结随机特征序列模型 MSE=1.051576。
- 模型报告：只报告训练分数：拒绝；超过调参预算：拒绝；未冻结随机种子：拒绝。
- 漂移响应：均值漂移 0.8 超过阈值 0.5，触发停止自动上线并重新校准。
- 校准：Brier 分数由 0.160 降至 0.025。
- 解释：特征重要性不等于因果；Top-2 Jaccard=0.333333，因果主张由独立门禁拒绝。
- 收益：毛收益 0.030；成本 0.006；成本后收益：0.024。
- 当前教学 notebook：`notebooks/lower/ch08_ml_alpha_model.ipynb`、`notebooks/lower/ch12_ml_alpha_validation.ipynb`、`notebooks/lower/ch13_ml_alpha_research.ipynb`；历史整路线 oracle 不再由独立 notebook 维护。
