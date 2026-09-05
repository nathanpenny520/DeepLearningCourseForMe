# 04 · Matrix Factorizations & SVD

矩阵分解与奇异值分解：LU、QR、Cholesky 三大分解，SVD 的定义与几何直觉，低秩近似与 Eckart-Young 定理，以及图像压缩、PCA、伪逆等应用。代码为主，配可视化。

## 前置要求

- 已完成 [01 Tensor Basics](../01-tensor-basics)、[02 Matrix Operations](../02-matrix-operations)、[03 Eigenvalues & Eigenvectors](../03-eigenvalues-eigenvectors)
- 关键前置：矩阵乘法、逆矩阵、行列式、特征值与特征向量、对称矩阵正交对角化

## 课程内容

| 章节 | 内容 |
|------|------|
| 0 | 环境配置与导入 |
| 1 | 为什么需要矩阵分解（动机与全景图） |
| 2 | LU 分解（PA=LU，高斯消元的矩阵形式） |
| 3 | QR 分解（A=QR，正交三角分解，Gram-Schmidt） |
| 4 | Cholesky 分解（A=LL^T，正定矩阵的平方根） |
| 5 | SVD 奇异值分解（A=UΣV^T，几何直觉） |
| 6 | SVD 与特征分解的关系（A^T A 的特征值 = 奇异值平方） |
| 7 | 低秩近似与 Eckart-Young 定理（截断 SVD） |
| 8 | 应用（图像压缩 / PCA 的 SVD 实现 / 伪逆） |
| - | 课后练习 |

## 学习建议

- SVD 是本节课的核心，第 5-8 节要连起来看
- 第 7 节的低秩近似是理解推荐系统、图像压缩、降维的关键
- 图像压缩例子需要实际运行看效果
