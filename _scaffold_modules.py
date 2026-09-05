# -*- coding: utf-8 -*-
"""Scaffold modules: probability, neural-networks, deep-learning-architectures."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

GUIDE = """## 环境配置

> 推荐 Python 3.12（不建议使用最新版如 3.14，深度学习库可能尚未兼容）。
> 以下命令均在**仓库根目录**下开始执行，先进入模块目录 `cd {name}`。
> 课程代码本身跨平台，仅安装和激活命令因系统而异。

**首选方案：uv**（极速 Python 包管理器，自动管理 Python 版本）。
传统 venv + pip 方案见文末备选。

---

### 方案一：uv（推荐）

#### 1. 安装 uv

| 系统 | 命令 |
|------|------|
| Mac | `brew install uv` |
| Windows | `winget install astral-sh.uv`（或 `pip install uv`） |

安装后验证：`uv --version`

#### 2. 创建虚拟环境（跨平台相同）

uv 会自动下载 Python 3.12，无需提前安装。

```bash
cd {name}
uv venv venv --python 3.12
```

#### 3. 激活虚拟环境

| 系统 | 命令 |
|------|------|
| Mac | `source venv/bin/activate` |
| Windows (PowerShell) | `venv\\Scripts\\Activate.ps1` |

> Windows 若提示执行策略禁止，先运行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

#### 4. 安装依赖（跨平台相同）

```bash
uv pip install -r requirements.txt
```

---

### 注册 Jupyter 内核（推荐执行一次，跨平台相同）

由于虚拟环境位于模块子目录，VS Code 自动搜索可能发现不了。注册后 VS Code 和网页版 Jupyter 都能直接选择。

```bash
# 确保已激活 venv
python -m ipykernel install --user --name={kernel} --display-name="Python ({kernel})"
```

> 每个模块只需注册一次。新增其他模块时用不同的 `--name` 重复即可。

---

### 在 VS Code 中使用 Notebook

打开 `{notebook}`：

1. 点击右上角内核选择器（Mac: `Cmd+Shift+P`，Windows: `Ctrl+Shift+P` → `Notebook: Select Notebook Kernel`）
2. 选 **Select Another Kernel...** → **Jupyter Kernel...**（不是 Python Environments...）
3. 选择 **Python ({kernel})**

> 关键区别：`Python Environments...` 是 VS Code 自动扫描的解释器，发现不了子目录的 venv；`Jupyter Kernel...` 是从注册表读取的，注册的内核在这里。

#### 如果列表里没有

1. 重载窗口：`Cmd+Shift+P` / `Ctrl+Shift+P` → `Developer: Reload Window`
2. 仍没有则手动指定：`Python Environments...` → `Enter interpreter path...` → 浏览到 `{name}/venv/bin/python`（Mac）或 `{name}\\venv\\Scripts\\python.exe`（Windows）

---

### 方案二：传统 venv + pip（备选）

#### 1. 安装 Python 3.12

| 系统 | 方式 |
|------|------|
| Mac | `brew install python@3.12` |
| Windows | 从 [python.org](https://www.python.org/downloads/) 下载安装，勾选 "Add Python to PATH" |

#### 2. 创建并激活虚拟环境

```bash
cd {name}
```

| 系统 | 创建 | 激活 |
|------|------|------|
| Mac | `python3.12 -m venv venv` | `source venv/bin/activate` |
| Windows | `python -m venv venv` | `venv\\Scripts\\Activate.ps1` |

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

安装完成后同样执行上面的「注册 Jupyter 内核」步骤。

---

### 网页版 Jupyter（可选）

注册内核后启动：

```bash
jupyter notebook
```

在网页中 **Kernel → Change Kernel** 选择 `Python ({kernel})`。

---

### 内核维护命令（跨平台相同）

```bash
# 查看已注册内核
jupyter kernelspec list

# 删除内核（仅移除注册记录，不删除 venv）
jupyter kernelspec remove {kernel}
```
"""

REQS = """# ============================================================
# {title} - Python 依赖
# Python 版本：3.12（推荐，详见 guide.md）
# 安装：uv pip install -r requirements.txt  或  pip install -r requirements.txt
# ============================================================

# --- 核心计算 ---

# PyTorch：张量计算 + 自动求导（本课程核心库，与 linear-algebra / calculus 模块同版本）
# Mac 版默认支持 MPS 加速；CUDA 用户请参考
# https://pytorch.org/get-started/locally/ 选择带 CUDA 的 wheel
torch==2.13.0

# NumPy：数值计算基础（与 torch 张量互转）
numpy==2.5.2

# Matplotlib：数据可视化
matplotlib==3.11.1

# --- Jupyter 运行环境 ---

# ipykernel：VS Code / Jupyter 运行 notebook 的 Python 内核（必需）
ipykernel==7.3.0

# jupyter：网页版 Jupyter Notebook 元包
# VS Code 用户可注释掉此行以减少安装体积（VS Code 直连 venv 内核，不需要网页版）
jupyter==1.1.1
"""


def readme_header(name, desc, lessons, kernel):
    rows = "\n".join(
        f"| {i:02d} | {t} | [{d}](./{d}) | ✅ |" for i, (t, d) in enumerate(lessons, 1)
    )
    tree = "\n".join(
        f"├── {d}/" if i < len(lessons) else f"└── {d}/"
        for i, (_, d) in enumerate(lessons)
    )
    return f"""# {name} 模块

{desc}每堂课一个独立子文件夹，按编号排序，共用一套依赖与虚拟环境。

## 课程目录

| # | 课程 | 文件夹 | 状态 |
|---|------|--------|------|
{rows}

## 环境配置

本模块所有课程共用一套依赖与虚拟环境，完整配置步骤见 [`guide.md`](./guide.md)。

快速上手（Mac，在仓库根目录下执行）：

```bash
cd {name}
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name={kernel} --display-name="Python ({kernel})"
```

> Windows 用户的激活命令及完整步骤见 [`guide.md`](./guide.md)。

然后在 VS Code 中打开对应课程的 `.ipynb`，内核选择器里选 **Python ({kernel})**。

## 模块结构

```
{name}/
├── README.md              # 本文件：模块总览与课程目录
├── guide.md               # 环境配置指南（全模块通用）
├── requirements.txt       # Python 依赖（全模块通用）
├── venv/                  # 虚拟环境（不提交）
{tree}
```
"""

MODULES = [
    dict(
        name="probability",
        kernel="probability",
        desc="深度学习所需的概率统计系列课程：分布、期望方差、条件概率与贝叶斯、独立性、极大似然、信息论、采样与蒙特卡洛，最终在深度学习场景中综合应用。",
        lessons=[
            ("Random Variables & Distributions（随机变量与分布：PMF/PDF/CDF）", "01-random-variables-distributions"),
            ("Expectation & Variance（期望与方差：线性性质、矩）", "02-expectation-variance"),
            ("Joint, Conditional & Bayes（联合分布、条件概率、贝叶斯定理）", "03-joint-conditional-bayes"),
            ("Independence & Covariance（独立性与协方差矩阵）", "04-independence-covariance"),
            ("Maximum Likelihood（极大似然估计与损失函数）", "05-maximum-likelihood"),
            ("Information Theory（信息论：熵、交叉熵、KL 散度）", "06-information-theory"),
            ("Sampling & Monte Carlo（采样与蒙特卡洛方法）", "07-sampling-monte-carlo"),
            ("Probability in Deep Learning（深度学习中的概率综合应用）", "08-probability-in-deep-learning"),
        ],
    ),
    dict(
        name="neural-networks",
        kernel="neural-networks",
        desc="深度学习核心系列课程：从零实现 MLP、激活与损失、训练循环、正则化、权重初始化、优化器实践、过拟合与泛化，最终搭出迷你训练框架。",
        lessons=[
            ("MLP from Scratch（从零实现多层感知机）", "01-mlp-from-scratch"),
            ("Activations & Losses（激活函数与损失函数）", "02-activations-losses"),
            ("Training Loop（训练循环：batch/epoch/验证集）", "03-training-loop"),
            ("Regularization（正则化：L1/L2/Dropout/早停）", "04-regularization"),
            ("Weight Initialization（权重初始化：Xavier/Kaiming 推导）", "05-weight-initialization"),
            ("Optimizers in Practice（优化器实践：SGD/动量/Adam）", "06-optimizers-in-practice"),
            ("Overfitting & Generalization（过拟合与泛化：偏差-方差）", "07-overfitting-generalization"),
            ("Mini Training Framework（综合：迷你训练框架）", "08-mini-training-framework"),
        ],
    ),
    dict(
        name="deep-learning-architectures",
        kernel="deep-learning-architectures",
        desc="深度学习架构系列课程：卷积神经网络基础与经典网络、注意力机制、Transformer 架构、GPT 语言模型，最后用全课程回顾把四块数学与工程拼成完整链路。",
        lessons=[
            ("CNN Basics（卷积神经网络基础：卷积/池化/感受野）", "01-cnn-basics"),
            ("CNN Classic Networks（经典网络：LeNet/VGG/ResNet）", "02-cnn-classic-networks"),
            ("Attention Mechanism（注意力机制：QKV/多头）", "03-attention-mechanism"),
            ("Transformer Architecture（Transformer 架构：Encoder/位置编码/掩码）", "04-transformer-architecture"),
            ("GPT & Language Models（GPT 与语言模型：自回归/训练/生成）", "05-gpt-language-models"),
            ("Full Course Review（全课程回顾：从线性代数到 Transformer）", "06-full-course-review"),
        ],
    ),
]


def main():
    for m in MODULES:
        d = os.path.join(ROOT, m["name"])
        os.makedirs(d, exist_ok=True)
        for _, lesson_dir in m["lessons"]:
            os.makedirs(os.path.join(d, lesson_dir), exist_ok=True)
        with open(os.path.join(d, "guide.md"), "w", encoding="utf-8") as fp:
            fp.write(GUIDE.format(name=m["name"], kernel=m["kernel"],
                                  notebook=m["lessons"][0][1] + "/" + m["lessons"][0][1] + ".ipynb"))
        with open(os.path.join(d, "requirements.txt"), "w", encoding="utf-8") as fp:
            fp.write(REQS.format(title=m["name"]))
        with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as fp:
            fp.write(readme_header(m["name"], m["desc"], m["lessons"], m["kernel"]))
        print("scaffolded:", m["name"])


if __name__ == "__main__":
    main()
