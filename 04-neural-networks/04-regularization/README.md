# Regularization

多项式过拟合演示、L1（稀疏）vs L2（收缩）、Dropout 原理与实现、早停与数据增强、AdamW 与权重衰减的关系。

**运行**：VS Code 打开 `regularization.ipynb`，内核选 **Python (neural-networks)**。

**前置**：先完成 [03 Training Loop](../03-training-loop)

**关键结论**：L1 稀疏 vs L2 收缩（几何直觉：L1 的角点）是高频考点；Dropout 期望不变的缩放 1/(1−p) 要会推导。
