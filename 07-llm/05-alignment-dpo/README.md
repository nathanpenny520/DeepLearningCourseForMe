# 05 Alignment：RLHF & DPO

对齐与偏好优化：RLHF 四阶段拆解、DPO 推导、极小机理演示（β 敏感性）。

**运行**：打开 `alignment-dpo.ipynb`，内核 **Python (llm)**。

**关键结论**：DPO 把 RLHF 消元成"偏好对 + 参考模型"的分类损失，免去 RM 与 PPO；β 控制拉开好坏差距的强度。
