# Attention Mechanism

QKV 检索类比、缩放点积注意力（手写 vs 官方对拍）、√d_k 的方差分析、注意力矩阵直觉、多头注意力实现。

**运行**：VS Code 打开 `attention-mechanism.ipynb`，内核选 **Python (deep-learning-architectures)**。

**前置**：建议对照 [probability 06](../03-probability/06-information-theory)

**关键结论**："除以 √d 保持方差 1"的推导是面试必考；多头 = 子空间分解，与 PCA 的多方向思想一脉相承。
