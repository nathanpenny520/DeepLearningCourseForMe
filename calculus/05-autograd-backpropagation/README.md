# 05 · Automatic Differentiation & Backpropagation

自动微分与反向传播：三种求导方式对比、计算图、前向/反向模式成本分析、从零手写迷你 autograd、两层 MLP 手写反向传播与 torch 对拍、VJP、常见坑。

## 前置要求

- 先完成 [01-derivatives](../01-derivatives) ~ [04-hessian-taylor-expansion](../04-hessian-taylor-expansion)
- 理解链式法则（01 课 3.4）是本节的地基

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 数值/符号/自动微分三对比 |
| 2 | 计算图：节点存值 + 局部导数 |
| 3 | 前向模式 vs 反向模式 |
| 4 | 手写迷你 autograd（70 行） |
| 5 | 两层 MLP 手写反向传播 |
| 6 | 向量-雅可比积 VJP |
| 7 | 常见坑：原地操作 / detach / 非标量 |
| - | 课后练习 |

## 学习建议

- 迷你 autograd 建议逐行读，它就是 torch.autograd 的最小骨架
- 5 节的手写 backward 是"会用框架"到"懂框架"的分水岭
