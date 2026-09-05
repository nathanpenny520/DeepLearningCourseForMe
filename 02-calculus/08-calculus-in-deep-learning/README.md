# 08 · Calculus in Deep Learning

深度学习中的微积分综合应用：激活函数导数与饱和、Softmax Jacobian、Softmax+交叉熵的极简梯度 ∂L/∂z = s−y、LayerNorm 导数、梯度消失/爆炸的微积分根源、log-sum-exp 数值稳定性、期望风险与 mini-batch。

## 前置要求

- 已完成 01-07 全部内容
- 关键前置：链式法则、Jacobian、Hessian、积分/期望

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 微积分在深度学习中的全景图 |
| 2 | 激活函数导数（sigmoid/tanh/ReLU） |
| 3 | Softmax 的 Jacobian |
| 4 | Softmax + 交叉熵：∂L/∂z = s − y |
| 5 | LayerNorm 的多元链式法则 |
| 6 | 梯度消失/爆炸的指数根源 |
| 7 | log-sum-exp 数值稳定性 |
| 8 | 期望风险与 mini-batch |
| - | 课后练习 |

## 学习建议

- 4 节 "$s-y$" 是全模块最重要的一个结果，推导务必独立写一遍
- 6 节与 01 课的链式法则、04 课的谱分析互证，值得连起来看
