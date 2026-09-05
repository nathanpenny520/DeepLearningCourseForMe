# Pretraining Practice

预训练实践：更大语料 + 训练 / 验证划分 + PPL 指标 + 过拟合观察，从玩具设置理解真实 LLM 预训练的全部骨架。

**运行**：VS Code 打开 `pretraining-practice.ipynb`，内核选 **Python (llm)**。

**关键结论**：next-token 自监督 + 交叉熵 + PPL = exp(loss)；训练 / 验证分叉是过拟合信号，也是早停依据。
