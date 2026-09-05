# Jacobian & Chain Rule

雅可比矩阵与向量链式法则：常见算子 Jacobian、局部线性化、Jacobian 连乘 = 反向传播的数学本质、向量-雅可比积（VJP）。

**运行**：VS Code 打开 `jacobian-chain-rule.ipynb`，内核选 **Python (calculus)**。

**前置**：先完成 [01](../01-derivatives) 与 [02](../02-multivariate-calculus-gradients)

**关键结论**：反向传播 = 从后往前乘 Jacobian，且"只乘向量不乘矩阵"（VJP）。
