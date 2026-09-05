# probability 模块

深度学习所需的概率统计系列课程：分布、期望方差、条件概率与贝叶斯、独立性、极大似然、信息论、采样与蒙特卡洛，最终在深度学习场景中综合应用。每堂课一个独立子文件夹，按编号排序，共用一套依赖与虚拟环境。

## 课程目录

| # | 课程 | 文件夹 | 状态 |
|---|------|--------|------|
| 01 | Random Variables & Distributions（随机变量与分布：PMF/PDF/CDF） | [01-random-variables-distributions](./01-random-variables-distributions) | ✅ |
| 02 | Expectation & Variance（期望与方差：线性性质、矩） | [02-expectation-variance](./02-expectation-variance) | ✅ |
| 03 | Joint, Conditional & Bayes（联合分布、条件概率、贝叶斯定理） | [03-joint-conditional-bayes](./03-joint-conditional-bayes) | ✅ |
| 04 | Independence & Covariance（独立性与协方差矩阵） | [04-independence-covariance](./04-independence-covariance) | ✅ |
| 05 | Maximum Likelihood（极大似然估计与损失函数） | [05-maximum-likelihood](./05-maximum-likelihood) | ✅ |
| 06 | Information Theory（信息论：熵、交叉熵、KL 散度） | [06-information-theory](./06-information-theory) | ✅ |
| 07 | Sampling & Monte Carlo（采样与蒙特卡洛方法） | [07-sampling-monte-carlo](./07-sampling-monte-carlo) | ✅ |
| 08 | Probability in Deep Learning（深度学习中的概率综合应用） | [08-probability-in-deep-learning](./08-probability-in-deep-learning) | ✅ |

## 环境配置

本模块所有课程共用一套依赖与虚拟环境，完整配置步骤见 [`guide.md`](./guide.md)。

快速上手（Mac，在仓库根目录下执行）：

```bash
cd probability
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=probability --display-name="Python (probability)"
```

> Windows 用户的激活命令及完整步骤见 [`guide.md`](./guide.md)。

然后在 VS Code 中打开对应课程的 `.ipynb`，内核选择器里选 **Python (probability)**。

## 模块结构

```
probability/
├── README.md              # 本文件：模块总览与课程目录
├── guide.md               # 环境配置指南（全模块通用）
├── requirements.txt       # Python 依赖（全模块通用）
├── venv/                  # 虚拟环境（不提交）
├── 01-random-variables-distributions/
├── 02-expectation-variance/
├── 03-joint-conditional-bayes/
├── 04-independence-covariance/
├── 05-maximum-likelihood/
├── 06-information-theory/
├── 07-sampling-monte-carlo/
├── 08-probability-in-deep-learning/
```
