# 线性代数模块

深度学习所需的线性代数系列课程，每堂课一个独立子文件夹，按编号排序。

## 课程目录

| # | 课程 | 文件夹 | 状态 |
|---|------|--------|------|
| 01 | Tensor Basics（张量操作入门，含自动求导简要拓展） | [01-tensor-basics](./01-tensor-basics) | ✅ |
| 02 | Matrix Operations（矩阵运算与线性变换，含 2D 可视化） | [02-matrix-operations](./02-matrix-operations) | ✅ |
| 03 | Eigenvalues & Eigenvectors（特征值与特征向量，含 PCA 入门） | [03-eigenvalues-eigenvectors](./03-eigenvalues-eigenvectors) | ✅ |
| 04 | Matrix Factorizations & SVD（矩阵分解与奇异值分解） | [04-matrix-factorizations-svd](./04-matrix-factorizations-svd) | ✅ |
| 05 | Vector Spaces, Rank & Linear Systems（向量空间、秩与线性方程组） | [05-vector-spaces-rank](./05-vector-spaces-rank) | ✅ |
| 06 | Projections, Least Squares & Orthogonalization（投影、最小二乘与正交化） | [06-projections-least-squares](./06-projections-least-squares) | ✅ |
| 07 | Positive Definite Matrices & Quadratic Forms（正定矩阵与二次型） | [07-positive-definite-quadratic](./07-positive-definite-quadratic) | ✅ |
| 08 | Linear Algebra in Deep Learning（深度学习中的线性代数综合应用） | [08-linear-algebra-in-deep-learning](./08-linear-algebra-in-deep-learning) | ✅ |

## 环境配置

本模块所有课程共用一套依赖与虚拟环境，完整配置步骤见 [`guide.md`](./guide.md)。

快速上手（Mac，在仓库根目录下执行）：

```bash
cd linear-algebra
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=linear-algebra --display-name="Python (linear-algebra)"
```

> Windows 用户的激活命令及完整步骤见 [`guide.md`](./guide.md)。

然后在 VS Code 中打开对应课程的 `.ipynb`，内核选择器里选 **Python (linear-algebra)**。

## 模块结构

```
01-linear-algebra/
├── README.md              # 本文件：模块总览与课程目录
├── guide.md               # 环境配置指南（全模块通用）
├── requirements.txt       # Python 依赖（全模块通用）
├── venv/                  # 虚拟环境（不提交）
├── 01-tensor-basics/      # 第 1 课
│   ├── README.md
│   └── tensor-basics.ipynb
├── 02-matrix-operations/  # 第 2 课
│   ├── README.md
│   └── matrix-operations.ipynb
├── 03-eigenvalues-eigenvectors/  # 第 3 课
│   ├── README.md
│   └── eigenvalues-eigenvectors.ipynb
├── 04-matrix-factorizations-svd/  # 第 4 课
│   ├── README.md
│   └── matrix-factorizations-svd.ipynb
├── 05-vector-spaces-rank/  # 第 5 课
│   ├── README.md
│   └── vector-spaces-rank.ipynb
├── 06-projections-least-squares/  # 第 6 课
│   ├── README.md
│   └── projections-least-squares.ipynb
├── 07-positive-definite-quadratic/  # 第 7 课
│   ├── README.md
│   └── positive-definite-quadratic.ipynb
└── 08-linear-algebra-in-deep-learning/  # 第 8 课
    ├── README.md
    └── linear-algebra-in-deep-learning.ipynb
```
