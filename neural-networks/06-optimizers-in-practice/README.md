# 06 · Optimizers in Practice

优化器实践：SGD/Momentum/RMSprop/Adam 横评、Adam 超参数、学习率调度（StepLR/Cosine）、warmup 与梯度裁剪、实操清单。

## 前置要求

- 先完成 [03-training-loop](../03-training-loop)
- calculus 06（梯度下降理论）对照

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 四种优化器更新规则与横评 |
| 2 | Adam 超参数 |
| 3 | 学习率调度 |
| 4 | Warmup 与梯度裁剪 |
| 5 | 实操清单 |
| - | 课后练习 |

## 学习建议

- 优化器横评要自己跑一遍，感受"收敛速度 vs 泛化"的取舍
- AdamW + warmup + 裁剪是大模型的标配，记住这个组合
