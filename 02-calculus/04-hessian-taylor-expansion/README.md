# Hessian & Taylor Expansion

多元二阶泰勒展开、凸性判据（正定→极小、不定→鞍点）、牛顿法推导、条件数与 GD 收敛速度、深度学习中的 Hessian-向量积（HVP）。

**运行**：VS Code 打开 `hessian-taylor-expansion.ipynb`，内核选 **Python (calculus)**。

**前置**：先完成 [01](../01-derivatives) ~ [03](../03-jacobian-chain-rule)

**关键结论**：Hessian 正定 → 局部极小；条件数大 → GD 慢，牛顿法用 H⁻¹ 矫正（与 linear-algebra 07 互证）。
