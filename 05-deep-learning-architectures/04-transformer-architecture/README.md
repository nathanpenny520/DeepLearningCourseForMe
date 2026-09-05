# 04 · Transformer Architecture

Transformer 架构：正弦位置编码（相对位置性质验证）、Pre-LN Block 实现、因果掩码、完整 Encoder 栈、LayerNorm vs BatchNorm。

## 前置要求

- 先完成 [03-attention-mechanism](../03-attention-mechanism)

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 位置编码：为什么需要、正弦公式 |
| 2 | Transformer Block（Pre-LN） |
| 3 | 因果掩码 |
| 4 | 完整 Encoder 栈 |
| 5 | LayerNorm vs BatchNorm |
| - | 课后练习 |

## 学习建议

- 位置编码"相对位置只依赖偏移"的性质是本课最有价值的验证实验
- Pre-LN 是现代大模型的主流选择，理解它与原始 Post-LN 的差异
