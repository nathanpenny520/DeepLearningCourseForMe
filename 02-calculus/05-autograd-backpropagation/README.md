# Automatic Differentiation & Backpropagation

数值 / 符号 / 自动微分三种方式对比、计算图、前向 / 反向模式成本分析、从零手写迷你 autograd、两层 MLP 手写反向传播与 torch 对拍。

**运行**：VS Code 打开 `autograd-backpropagation.ipynb`，内核选 **Python (calculus)**。

**前置**：先完成 [01](../01-derivatives) ~ [04](../04-hessian-taylor-expansion)

**关键结论**：迷你 autograd 就是 torch.autograd 的最小骨架；手写 backward 是"会用框架"到"懂框架"的分水岭。
