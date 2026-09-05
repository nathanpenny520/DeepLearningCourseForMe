# 07 Text-to-Image

文生图原理：Stable Diffusion 四组件拆解（VAE/U-Net/文本编码器/调度器）、交叉注意力实现与可视化、无分类器引导（CFG）。

**运行**：打开 `text-to-image.ipynb`，内核 **Python (computer-vision)**。

**关键结论**：文生图 = 条件扩散 + 潜空间压缩 + 引导；交叉注意力是文本注入图像的机制；CFG 用外推增强条件强度。
