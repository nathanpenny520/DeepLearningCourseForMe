# deep-learning-architectures 模块

深度学习架构系列课程：卷积神经网络基础与经典网络、注意力机制、Transformer 架构、GPT 语言模型，最后用全课程回顾把四块数学与工程拼成完整链路。每堂课一个独立子文件夹，按编号排序，共用一套依赖与虚拟环境。

## 课程目录

| # | 课程 | 文件夹 | 状态 |
|---|------|--------|------|
| 01 | CNN Basics（卷积神经网络基础：卷积/池化/感受野） | [01-cnn-basics](./01-cnn-basics) | ✅ |
| 02 | CNN Classic Networks（经典网络：LeNet/VGG/ResNet） | [02-cnn-classic-networks](./02-cnn-classic-networks) | ✅ |
| 03 | Attention Mechanism（注意力机制：QKV/多头） | [03-attention-mechanism](./03-attention-mechanism) | ✅ |
| 04 | Transformer Architecture（Transformer 架构：Encoder/位置编码/掩码） | [04-transformer-architecture](./04-transformer-architecture) | ✅ |
| 05 | GPT & Language Models（GPT 与语言模型：自回归/训练/生成） | [05-gpt-language-models](./05-gpt-language-models) | ✅ |
| 06 | Full Course Review（全课程回顾：从线性代数到 Transformer） | [06-full-course-review](./06-full-course-review) | ✅ |

## 环境配置

本模块所有课程共用一套依赖与虚拟环境，完整配置步骤见 [`guide.md`](./guide.md)。

快速上手（Mac，在仓库根目录下执行）：

```bash
cd deep-learning-architectures
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=deep-learning-architectures --display-name="Python (deep-learning-architectures)"
```

> Windows 用户的激活命令及完整步骤见 [`guide.md`](./guide.md)。

然后在 VS Code 中打开对应课程的 `.ipynb`，内核选择器里选 **Python (deep-learning-architectures)**。

## 模块结构

```
05-deep-learning-architectures/
├── README.md              # 本文件：模块总览与课程目录
├── guide.md               # 环境配置指南（全模块通用）
├── requirements.txt       # Python 依赖（全模块通用）
├── venv/                  # 虚拟环境（不提交）
├── 01-cnn-basics/
├── 02-cnn-classic-networks/
├── 03-attention-mechanism/
├── 04-transformer-architecture/
├── 05-gpt-language-models/
├── 06-full-course-review/
```
