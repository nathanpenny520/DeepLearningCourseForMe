# 微积分模块

深度学习所需的微积分系列课程，每堂课一个独立子文件夹，按编号排序。从导数第一性原理出发，经多元微积分、梯度 / Jacobian / Hessian、链式法则与反向传播、梯度下降与优化，最终落到深度学习场景中的综合应用，并为概率统计模块搭好积分的桥。

## 课程目录

| # | 课程 | 文件夹 | 状态 |
|---|------|--------|------|
| 01 | Derivatives（导数：极限定义、求导法则证明、数值微分、梯度下降入门） | [01-derivatives](./01-derivatives) | ✅ |
| 02 | Multivariate Calculus & Gradients（多元微积分与梯度向量） | — | 规划中 |
| 03 | Jacobian & Chain Rule（雅可比矩阵与向量链式法则） | — | 规划中 |
| 04 | Hessian & Taylor Expansion（Hessian 矩阵与二阶逼近） | — | 规划中 |
| 05 | Automatic Differentiation & Backprop（自动微分与反向传播） | — | 规划中 |
| 06 | Gradient Descent & Optimization（梯度下降、动量与学习率） | — | 规划中 |
| 07 | Integration & Probability Bridge（积分入门与概率统计衔接） | — | 规划中 |
| 08 | Calculus in Deep Learning（深度学习中的微积分综合应用） | — | 规划中 |

## 环境配置

本模块所有课程共用一套依赖与虚拟环境，完整配置步骤见 [`guide.md`](./guide.md)。

快速上手（Mac，在仓库根目录下执行）：

```bash
cd calculus
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=calculus --display-name="Python (calculus)"
```

> Windows 用户的激活命令及完整步骤见 [`guide.md`](./guide.md)。

然后在 VS Code 中打开对应课程的 `.ipynb`，内核选择器里选 **Python (calculus)**。

## 模块结构

```
calculus/
├── README.md              # 本文件：模块总览与课程目录
├── guide.md               # 环境配置指南（全模块通用）
├── requirements.txt       # Python 依赖（全模块通用）
├── venv/                  # 虚拟环境（不提交）
└── 01-derivatives/        # 第 1 课
    ├── README.md
    └── derivatives.ipynb
```
