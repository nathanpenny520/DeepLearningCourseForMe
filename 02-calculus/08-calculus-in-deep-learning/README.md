# Calculus in Deep Learning

深度学习中的微积分综合应用：激活函数导数与饱和、Softmax Jacobian、∂L/∂z = s−y、LayerNorm 导数、梯度消失 / 爆炸、log-sum-exp 数值稳定性。

**运行**：VS Code 打开 `calculus-in-deep-learning.ipynb`，内核选 **Python (calculus)**。

**前置**：先完成 [01](../01-derivatives) ~ [07](../07-integration-probability)

**关键结论**："s − y" 是分类任务最核心的梯度结果，务必独立推导；梯度消失 / 爆炸的指数根源解释深网难训。
