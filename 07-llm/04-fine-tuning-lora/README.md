# Fine-tuning & LoRA

微调与低秩适配：预训练 TinyGPT → 倒装语料风格迁移，对比全量微调 vs LoRA（1.5% 参数逼近全量效果）。

**运行**：VS Code 打开 `fine-tuning-lora.ipynb`，内核选 **Python (llm)**。

**关键结论**：W' = W + (α/r)BA，B 零初始化保底座；低秩增量够用（intrinsic dimension 小）；LoRA 是开源生态定制标配。
