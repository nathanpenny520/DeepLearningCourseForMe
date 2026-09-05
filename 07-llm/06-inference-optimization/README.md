# Inference Optimization

推理优化：KV Cache 计时实验（O(T) vs O(T²)）、对称量化实现与误差测量、Llama-7B KV 显存估算。

**运行**：VS Code 打开 `inference-optimization.ipynb`，内核选 **Python (llm)**。

**关键结论**：KV Cache 是自回归推理标配；量化让 7B 模型可部署；GQA / 投机解码 / 连续批处理是生产级进阶。
