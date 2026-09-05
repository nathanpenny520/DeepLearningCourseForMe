# 03 · Training Loop

训练循环规范：train/val/test 划分、mini-batch 与 shuffle 的影响、完整训练骨架（zero_grad→backward→step）、常见问题速查。

## 前置要求

- 先完成 [01-mlp-from-scratch](../01-mlp-from-scratch) 与 [02-activations-losses](../02-activations-losses)

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 数据划分 |
| 2 | mini-batch 与 shuffle |
| 3 | 完整训练骨架 |
| 4 | 常见问题速查 |
| - | 课后练习 |

## 学习建议

- 把第 3 节的 `train` 函数保存下来，后续所有实验都基于它扩展
- "zero_grad 忘写"是新手最常见的 bug，记住梯度是累加的
