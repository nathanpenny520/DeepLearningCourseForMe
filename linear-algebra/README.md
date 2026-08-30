# 深度学习中的线性代数：PyTorch 实战

面向初学者的线性代数 + PyTorch 入门教学代码。边写边讲，在实操中理解张量、广播、自动求导等核心概念。

## 前置要求

- [uv](https://docs.astral.sh/uv/)（推荐，极速 Python 包管理器）或 Python 3.12 + pip
- 基础的 Python 语法知识（不需要精通，边写边学）

## 环境配置

详见 [`guide.md`](./guide.md)，包含 uv / 传统 venv 两种方案、VS Code 内核选择等完整步骤。

快速上手（uv）：

```bash
cd linear-algebra
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt
```

然后在 VS Code 中打开 `linearalgebra.ipynb`，内核选择器里选 `venv/bin/python` 即可（无需注册内核）。

## 课程内容

| 章节 | 内容 |
|---|---|
| 1 | 张量基础概念（标量、向量、矩阵、多维张量） |
| 2 | 张量创建方法（tensor / zeros / ones / rand / randn / empty） |
| 3 | 数据类型 dtype 与类型转换 |
| 4 | 统计量入门（mean / std，rand vs randn） |
| 5 | 形状操作（reshape / unsqueeze / squeeze / -1 自动推导） |
| 6 | 广播机制 Broadcasting |
| 7 | 原地操作 In-place（add_ / sub_ / mul_ / div_） |
| 8 | 拼接与堆叠（cat / stack） |
| 9 | 按维度统计（sum / dim / keepdim） |
| 10 | 自动求导入门（requires_grad / backward） |

## 学习建议

- 不要只看视频，时间充足一定要亲手敲一遍代码
- 时间不够，课后作业也务必独立完成
- 遇到环境问题先查 `guide.md`，也可以咨询 AI

## 文件结构

```
linear-algebra/
├── README.md
├── guide.md              # 环境配置指南
├── requirements.txt      # Python 依赖
└── linearalgebra.ipynb   # 教学代码（Jupyter Notebook）
```
