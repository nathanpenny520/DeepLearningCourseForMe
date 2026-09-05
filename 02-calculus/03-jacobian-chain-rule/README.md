# 03 · Jacobian & Chain Rule

雅可比矩阵与向量链式法则：多输入多输出函数的导数矩阵、常见算子 Jacobian、局部线性化、Jacobian 连乘 = 反向传播的数学本质、向量-雅可比积（VJP）。

## 前置要求

- 先完成 [01-derivatives](../01-derivatives)、[02-multivariate-calculus-gradients](../02-multivariate-calculus-gradients)
- 会用矩阵乘法（linear-algebra 02）

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | Jacobian 定义与数值/解析对照 |
| 2 | 常见算子（线性/逐元素/二次型）的 Jacobian |
| 3 | 局部线性化：f(x+Δx) ≈ f(x) + J·Δx |
| 4 | 向量链式法则：J_z = J_g·J_f |
| 5 | 反向传播 = 从后往前乘 Jacobian |
| 6 | 向量-雅可比积（VJP） |
| 7 | 与线性代数的呼应 |
| - | 课后练习 |

## 学习建议

- 第 5 节是理解 autograd 的关键：记住"反向传播只乘向量，不乘矩阵"
- VJP 的数值验证务必自己跑一遍
