# Transformer Architecture

正弦位置编码（相对位置性质验证）、Pre-LN Block 实现、因果掩码、完整 Encoder 栈、LayerNorm vs BatchNorm。

**运行**：VS Code 打开 `transformer-architecture.ipynb`，内核选 **Python (deep-learning-architectures)**。

**前置**：先完成 [03 Attention Mechanism](../03-attention-mechanism)

**关键结论**：位置编码"相对位置只依赖偏移"是最有价值的验证实验；Pre-LN 是现代大模型的主流选择。
