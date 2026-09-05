# 08 · Mini Training Framework

综合：手写 Dataset/DataLoader/Trainer，同一框架训练 MLP 与线性模型、换损失函数、加正则，回顾"从数学到框架"的完整链路。

## 前置要求

- 完成本模块 01-07 全部课程

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | Dataset 与 DataLoader |
| 2 | Trainer 训练器 |
| 3 | 同一 Trainer 训练不同模型 |
| 4 | 扩展点：换损失/加正则 |
| 5 | 从数学到框架的完整链路 |
| - | 课后练习 |

## 学习建议

- 本课是 nn-core 的"毕业设计"：把前七课全部装进一个可复用框架
- 对比 `torch.utils.data` 的官方实现，理解每个抽象层解决什么问题
