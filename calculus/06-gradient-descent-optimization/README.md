# 06 · Gradient Descent & Optimization

梯度下降与优化：一维收敛条件推导、学习率调度、SGD 噪声几何、动量、自适应方法（AdaGrad/RMSProp/Adam）、停止条件与深度学习实操经验。

## 前置要求

- 先完成 [01-derivatives](../01-derivatives) ~ [05-autograd-backpropagation](../05-autograd-backpropagation)

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 收敛条件：α < 2/λ_max |
| 2 | 学习率调度（固定/衰减/余弦） |
| 3 | SGD：无偏噪声梯度 |
| 4 | 动量：速度累积 |
| 5 | AdaGrad / RMSProp / Adam |
| 6 | 停止条件 |
| 7 | 实操经验（warmup/线性缩放/clip） |
| - | 课后练习 |

## 学习建议

- 1 节的推导看懂后，学习率的所有直觉都有了数学根据
- 5 节的收敛曲线对比是"为什么默认用 Adam"的最好证据
