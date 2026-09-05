# 02 Scaling Laws

规模定律：词级 Zipf 语料上实测损失随参数量/数据量的幂律下降，拟合幂律指数，理解 Chinchilla 的"20 token/参数"配比。

**运行**：打开 `scaling-laws.ipynb`，内核 **Python (llm)**。

**关键结论**：L(N)、L(D) 在 log-log 坐标近似直线；数据量驱动最直观（9.1→3.7）；参数与数据须按比例增长。
