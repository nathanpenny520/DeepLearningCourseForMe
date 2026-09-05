# Deep Learning Course For Me

个人深度学习系列课程笔记与实战代码。按模块组织，每个模块下含多堂编号课程，共用一套依赖与虚拟环境。

## 课程目录

| 模块 | 说明 | 文件夹 | 课程数 |
|------|------|--------|--------|
| 线性代数 | 张量、矩阵运算、特征值、SVD 等 | [linear-algebra](./linear-algebra) | 8 |
| 微积分 | 导数、多元微积分、梯度、链式法则、自动微分等 | [calculus](./calculus) | 8 |

各模块内的具体课程目录见对应模块的 `README.md`。

## 快速开始

```bash
# 1. 克隆仓库
git clone git@github.com:nathanpenny520/DeepLearningCourseForMe.git
cd DeepLearningCourseForMe

# 2. 进入想学习的模块，按其 guide.md 配置环境
cd linear-algebra
# 完整步骤见 linear-algebra/guide.md，核心命令：
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=linear-algebra --display-name="Python (linear-algebra)"

# 3. 在 VS Code 中打开对应课程的 .ipynb，内核选择 Python (linear-algebra)
```

## 仓库结构

```
DeepLearningCourseForMe/
├── README.md                 # 本文件：模块总览
├── .gitignore
├── linear-algebra/           # 模块：线性代数
│   ├── README.md             # 模块总览与课程目录
│   ├── guide.md              # 环境配置（模块内通用）
│   ├── requirements.txt      # 依赖（模块内通用，已锁定版本）
│   ├── venv/                 # 虚拟环境（不提交）
│   └── 01-tensor-basics/     # 第 1 课
│       ├── README.md         # 本课说明
│       └── tensor-basics.ipynb
└── calculus/                 # 模块：微积分
    ├── README.md             # 模块总览与课程目录
    ├── guide.md              # 环境配置（模块内通用）
    ├── requirements.txt      # 依赖（模块内通用，已锁定版本）
    ├── venv/                 # 虚拟环境（不提交）
    └── 01-derivatives/       # 第 1 课
        ├── README.md
        └── derivatives.ipynb
```

每个模块共用一套虚拟环境；进入模块目录后按其 `guide.md` 配置，再打开对应课程的 notebook。

## 通用环境要求

- Python 3.12
- 推荐包管理器：[uv](https://docs.astral.sh/uv/)（也支持传统 venv + pip）
- 编辑器：VS Code + Jupyter 扩展（或网页版 Jupyter Notebook）
