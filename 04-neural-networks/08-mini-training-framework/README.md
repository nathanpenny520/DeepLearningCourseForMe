# Mini Training Framework

综合收尾：手写 Dataset / DataLoader / Trainer，同一框架训练 MLP 与线性模型、换损失函数、加正则，回顾"从数学到框架"的完整链路。

**运行**：VS Code 打开 `mini-training-framework.ipynb`，内核选 **Python (neural-networks)**。

**前置**：先完成本模块 [01](../01-mlp-from-scratch) ~ [07](../07-overfitting-generalization)

**关键结论**：本课是 nn-core 的"毕业设计"：把前七课装进一个可复用框架；对照 torch.utils.data 官方实现理解抽象层。
