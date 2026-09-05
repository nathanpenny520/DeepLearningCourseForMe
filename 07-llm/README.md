# llm 模块

大语言模型系列课程：BPE 分词、规模定律与数据、预训练实践、微调与 LoRA、对齐（RLHF/DPO）、推理优化（KV Cache/量化）、RAG 与智能体、评估与安全。每堂课一个独立子文件夹，按编号排序，共用一套依赖与虚拟环境。

## 课程目录

| # | 课程 | 文件夹 | 状态 |
|---|------|--------|------|
| 01 | Tokenization（BPE 分词：从字符到子词） | [01-tokenization](./01-tokenization) | ✅ |
| 02 | Scaling Laws（规模定律与数据配比） | [02-scaling-laws](./02-scaling-laws) | ✅ |
| 03 | Pretraining Practice（预训练实践：语料/上下文/损失） | [03-pretraining-practice](./03-pretraining-practice) | ✅ |
| 04 | Fine-tuning & LoRA（微调与低秩适配） | [04-fine-tuning-lora](./04-fine-tuning-lora) | ✅ |
| 05 | Alignment：RLHF & DPO（对齐与偏好优化） | [05-alignment-dpo](./05-alignment-dpo) | ✅ |
| 06 | Inference Optimization（推理优化：KV Cache 与量化） | [06-inference-optimization](./06-inference-optimization) | ✅ |
| 07 | RAG & Agents（检索增强与智能体） | [07-rag-agents](./07-rag-agents) | ✅ |
| 08 | Evaluation & Safety（评估、幻觉与安全） | [08-evaluation-safety](./08-evaluation-safety) | ✅ |

## 环境配置

本模块所有课程共用一套依赖与虚拟环境，完整配置步骤见 [`guide.md`](./guide.md)。

快速上手（Mac，在仓库根目录下执行）：

```bash
cd llm
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=llm --display-name="Python (llm)"
```

> Windows 用户的激活命令及完整步骤见 [`guide.md`](./guide.md)。

然后在 VS Code 中打开对应课程的 `.ipynb`，内核选择器里选 **Python (llm)**。

## 模块结构

```
07-llm/
├── README.md              # 本文件：模块总览与课程目录
├── guide.md               # 环境配置指南（全模块通用）
├── requirements.txt       # Python 依赖（全模块通用）
├── venv/                  # 虚拟环境（不提交）
├── 01-tokenization/
├── 02-scaling-laws/
├── 03-pretraining-practice/
├── 04-fine-tuning-lora/
├── 05-alignment-dpo/
├── 06-inference-optimization/
├── 07-rag-agents/
├── 08-evaluation-safety/
```
