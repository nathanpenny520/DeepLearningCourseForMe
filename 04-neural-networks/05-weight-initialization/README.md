# Weight Initialization

全零初始化的对称性陷阱、方差传播推导、Xavier（2/(d_in+d_out)）、Kaiming（2/d_in）、激活方差随层数传播的可视化。

**运行**：VS Code 打开 `weight-initialization.ipynb`，内核选 **Python (neural-networks)**。

**前置**：先完成 [01 MLP from Scratch](../01-mlp-from-scratch)

**关键结论**：记一条 Var[h] = d_in·Var[W]·Var[x]，其余都是推导；初始化与激活要匹配（Xavier↔tanh、Kaiming↔ReLU）。
