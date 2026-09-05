# Optimizers in Practice

SGD / Momentum / RMSprop / Adam 横评、Adam 超参数、学习率调度（StepLR / Cosine）、warmup 与梯度裁剪、实操清单。

**运行**：VS Code 打开 `optimizers-in-practice.ipynb`，内核选 **Python (neural-networks)**。

**前置**：先完成 [03 Training Loop](../03-training-loop)

**关键结论**：自己跑一遍优化器横评，体会"收敛速度 vs 泛化"的取舍；AdamW + warmup + 裁剪是大模型标配组合。
