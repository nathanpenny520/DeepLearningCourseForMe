# 线性代数模块

深度学习所需的线性代数系列课程，每堂课一个独立子文件夹，按编号排序。

## 课程目录

| # | 课程 | 文件夹 | 状态 |
|---|------|--------|------|
| 01 | Tensor Basics（张量操作入门，含自动求导简要拓展） | [01-tensor-basics](./01-tensor-basics) | ✅ |

## 环境配置

本模块所有课程共用一套依赖与虚拟环境，完整配置步骤见 [`guide.md`](./guide.md)。

快速上手（在仓库根目录下执行）：

```bash
cd linear-algebra
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt

# 注册内核（执行一次，VS Code 即可在列表中直接选择）
python -m ipykernel install --user --name=linear-algebra --display-name="Python (linear-algebra)"
```

然后在 VS Code 中打开对应课程的 `.ipynb`，内核选择器里选 **Python (linear-algebra)**。

## 模块结构

```
linear-algebra/
├── README.md              # 本文件：模块总览与课程目录
├── guide.md               # 环境配置指南（全模块通用）
├── requirements.txt       # Python 依赖（全模块通用）
├── venv/                  # 虚拟环境（不提交）
└── 01-tensor-basics/      # 第 1 课
    ├── README.md          # 本课说明
    └── tensor-basics.ipynb
```
