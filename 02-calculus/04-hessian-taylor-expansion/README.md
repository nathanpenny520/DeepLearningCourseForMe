# 04 · Hessian & Taylor Expansion

Hessian 矩阵与二阶逼近：多元二阶泰勒、凸性判据（正定→极小、不定→鞍点）、牛顿法推导、条件数与 GD 收敛速度、深度学习中的 Hessian-向量积（HVP）。

## 前置要求

- 先完成 [01-derivatives](../01-derivatives) ~ [03-jacobian-chain-rule](../03-jacobian-chain-rule)
- 建议先复习 linear-algebra 07 正定矩阵

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | Hessian 定义、克莱罗对称性 |
| 2 | 多元二阶泰勒展开与误差阶数 |
| 3 | Hessian 与凸性：极小/极大/鞍点判据 |
| 4 | 牛顿法：Δx = −H⁻¹∇f |
| 5 | 条件数与梯度下降速度 |
| 6 | 深度学习中的 Hessian：HVP 技巧 |
| - | 课后练习 |

## 学习建议

- 3 节的判据表与 linear-algebra 07 完全互通，值得对照看
- 牛顿法代码是理解"二阶方法为什么快"的最小例子
