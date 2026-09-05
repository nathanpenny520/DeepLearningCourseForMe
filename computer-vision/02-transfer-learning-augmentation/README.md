# 02 Transfer Learning & Augmentation

迁移学习与数据增强：源任务（横竖条）预训练 → 目标任务（对角线+噪声，16 张）上比较从零/微调/线性探针/增强四种方案。

**运行**：打开 `transfer-learning-augmentation.ipynb`，内核 **Python (computer-vision)**。

**关键结论**：预训练低层特征可复用；数据极小时线性探针更稳、微调上限更高；数据增强是"免费样本"。
