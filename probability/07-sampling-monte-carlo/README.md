# 07 · Sampling & Monte Carlo

采样与蒙特卡洛：MC 期望估计、逆变换采样（指数分布）、拒绝采样（Beta）、重要性采样、Metropolis-Hastings MCMC、深度学习中的采样应用。

## 前置要求

- 先完成 [02-expectation-variance](../02-expectation-variance)
- 建议复习 calculus 07（数值积分）

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 蒙特卡洛期望估计 |
| 2 | 逆变换采样：F⁻¹(U) |
| 3 | 拒绝采样 |
| 4 | 重要性采样 |
| 5 | Metropolis-Hastings MCMC |
| 6 | 深度学习中的采样 |
| - | 课后练习 |

## 学习建议

- 逆变换采样的证明（P(F⁻¹(U)≤x)=F(x)）务必自己推一遍
- 重参数化技巧是生成模型的关键，记住"采样=确定性变换+噪声"这个形式
