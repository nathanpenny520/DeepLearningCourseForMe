# 05 · Weight Initialization

权重初始化：全零对称性陷阱、方差传播推导、Xavier（2/(d_in+d_out)）、Kaiming（2/d_in）、激活方差随层数传播的可视化。

## 前置要求

- 先完成 [01-mlp-from-scratch](../01-mlp-from-scratch)
- 概率 02（方差）与 calculus 08（梯度消失）对照

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 全零初始化对称性 |
| 2 | 方差传播与 Xavier |
| 3 | Kaiming（ReLU 修正） |
| 4 | 与梯度消失的关联 |
| 5 | torch 默认初始化 |
| - | 课后练习 |

## 学习建议

- 方差传播公式只需记住"Var[h] = d_in·Var[W]·Var[x]"这一条，其余都是推导
- 初始化与激活匹配（Xavier↔tanh、Kaiming↔ReLU）是工程要点
