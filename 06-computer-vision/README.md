# computer-vision 模块

图像识别与生成的前沿系列课程：ViT 与现代识别、迁移学习与数据增强、检测与分割、GAN、VAE、扩散模型（DDPM）、文生图（Stable Diffusion）原理、前沿回顾与评估。每堂课一个独立子文件夹，按编号排序，共用一套依赖与虚拟环境。

## 课程目录

| # | 课程 | 文件夹 | 状态 |
|---|------|--------|------|
| 01 | ViT & Modern Recognition（Vision Transformer 与现代识别） | [01-vit-modern-recognition](./01-vit-modern-recognition) | ✅ |
| 02 | Transfer Learning & Augmentation（迁移学习与数据增强） | [02-transfer-learning-augmentation](./02-transfer-learning-augmentation) | ✅ |
| 03 | Detection & Segmentation（目标检测与语义分割入门） | [03-detection-segmentation](./03-detection-segmentation) | ✅ |
| 04 | GAN（对抗生成网络：原理与训练） | [04-gan](./04-gan) | ✅ |
| 05 | VAE（变分自编码器：重参数化与 ELBO） | [05-vae](./05-vae) | ✅ |
| 06 | Diffusion Models（扩散模型 DDPM：加噪与去噪） | [06-diffusion-models](./06-diffusion-models) | ✅ |
| 07 | Text-to-Image（文生图原理：Stable Diffusion 拆解） | [07-text-to-image](./07-text-to-image) | ✅ |
| 08 | CV Frontier Review（前沿回顾：自监督/评估指标） | [08-cv-frontier](./08-cv-frontier) | ✅ |

## 环境配置

本模块所有课程共用一套依赖与虚拟环境，完整配置步骤见 [`guide.md`](./guide.md)。

快速上手（Mac，在仓库根目录下执行）：

```bash
cd computer-vision
uv venv venv --python 3.12
source venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name=computer-vision --display-name="Python (computer-vision)"
```

> Windows 用户的激活命令及完整步骤见 [`guide.md`](./guide.md)。

然后在 VS Code 中打开对应课程的 `.ipynb`，内核选择器里选 **Python (computer-vision)**。

## 模块结构

```
06-computer-vision/
├── README.md              # 本文件：模块总览与课程目录
├── guide.md               # 环境配置指南（全模块通用）
├── requirements.txt       # Python 依赖（全模块通用）
├── venv/                  # 虚拟环境（不提交）
├── 01-vit-modern-recognition/
├── 02-transfer-learning-augmentation/
├── 03-detection-segmentation/
├── 04-gan/
├── 05-vae/
├── 06-diffusion-models/
├── 07-text-to-image/
├── 08-cv-frontier/
```
