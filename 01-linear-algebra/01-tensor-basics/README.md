# 01 · Tensor Basics

PyTorch 张量操作入门：从标量到多维张量，覆盖创建、形状操作、广播、原地操作、拼接堆叠、按维统计，末尾附自动求导（autograd）简要拓展。

## 前置要求

- Python 3.12
- 已按 [`../guide.md`](../guide.md) 配置好虚拟环境并安装依赖
- 基础 Python 语法知识（不需要精通，边写边学）

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 张量基础概念（标量、向量、矩阵、多维张量） |
| 2 | 张量创建方法（tensor / zeros / ones / rand / randn / empty） |
| 3 | 数据类型 dtype 与类型转换 |
| 4 | 统计量入门（mean / std，rand vs randn） |
| 5 | 形状操作（reshape / unsqueeze / squeeze / -1 自动推导） |
| 6 | 广播机制 Broadcasting |
| 7 | 原地操作 In-place（add_ / sub_ / mul_ / div_） |
| 8 | 拼接与堆叠（cat / stack） |
| 9 | 按维度统计（sum / dim / keepdim） |
| 10 | 自动求导入门（requires_grad / backward）— 简要拓展 |

## 学习建议

- 不要只看，时间充足一定要亲手敲一遍代码
- 时间不够，课后练习也务必独立完成
- 遇到环境问题先查 [`../guide.md`](../guide.md)
