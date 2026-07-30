# %% [markdown]
# # 模型与表示：从基线到 PyTorch、Transformer 与文本适配
#
# **研究目标。** 在固定小样本上学习 tensor、autograd、Dataset/DataLoader、
# 训练循环、CPU device、随机种子和 checkpoint，并比较线性/树/boosting、MLP、
# 因果卷积、RNN、Attention、Transformer 与文本适配边界。
# **假设。** 合成目标无噪声；序列长度固定；本地文本编码器只是冻结“预训练表示”的
# 最小代理，不宣称具备基础模型能力。
# **手算 oracle。** 时间均值对序列反转不变；最后差分卷积对 `[1,2,4,8]`
# 与反序列必然不同。
# **失败注入。** 时间置换若不改变“序列模型”输出，则该实现没有证明保留顺序。

# %%
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from math_for_quant.lower.ml_alpha_library import cross_check_classical_models
from math_for_quant.lower.ml_alpha_models import TorchTrainingConfig, sequence_order_sensitivity, train_tiny_mlp
from math_for_quant.lower.ml_alpha_text import compare_text_adaptation
from math_for_quant.lower.notebook_evidence import assert_expected, load_oracle_and_fixture


def main(oracle_path: Path) -> int:
    oracle, fixture = load_oracle_and_fixture(oracle_path)
    artifact = train_tiny_mlp(
        np.asarray(fixture["features"], dtype=np.float32),
        np.asarray(fixture["target"], dtype=np.float32),
        config=TorchTrainingConfig(
            int(fixture["seed"]), int(fixture["epochs"]), float(fixture["learning_rate"])
        ),
    )
    sensitivity = sequence_order_sensitivity(
        np.asarray(fixture["sequence"], dtype=np.float32), seed=int(fixture["sequence_seed"])
    )
    classical = cross_check_classical_models(
        np.asarray(fixture["classical_features"]), np.asarray(fixture["classical_target"]),
        np.asarray(fixture["classical_evaluation"]), boosting_rounds=int(fixture["boosting_rounds"]),
    )
    text = compare_text_adaptation(
        train_texts=fixture["train_texts"], train_labels=np.asarray(fixture["train_labels"]),
        inference_texts=fixture["inference_texts"], seed=int(fixture["text_seed"]),
    )
    plt.figure(figsize=(5, 2.5)); plt.bar(list(sensitivity)[1:], list(sensitivity.values())[1:]); plt.close()
    observed = {
        "mlp_loss": artifact.loss,
        "mean_pool_order_gap": sensitivity["mean_pool"],
        "causal_conv_order_gap": sensitivity["causal_conv"],
        "rnn_order_gap": sensitivity["rnn"],
        "attention_order_gap": sensitivity["attention"],
        "transformer_order_gap": sensitivity["transformer"],
        "stump_library_gap": classical.stump_max_gap,
        "boosting_library_gap": classical.boosting_max_gap,
        "token_count": text.token_count,
        "peft_parameter_ratio": text.peft_trainable_parameters / text.full_parameters,
    }
    assert_expected(observed, oracle)
    print("ml-alpha-model=passed " + " ".join(f"{key}={value:.6f}" for key, value in observed.items()))
    return 0


# %% [markdown]
# **敏感性。** 顺序差值同时报告四种架构，避免只展示一个“漂亮”模型。
# **限制。** 小型冻结编码器用于讲清微调、零/少样本与 PEFT 参数边界；生产 NLP/LLM
# 仍需模型卡、许可、tokenizer 版本、算力预算和发布时间审计。

# %%
if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/ml-alpha-model/oracle.json")))
