# Diffusion Models

扩散模型 DDPM：前向加噪闭式公式、噪声预测训练目标、反向去噪采样，完整实现并可视化加噪 / 去噪路径。

**运行**：VS Code 打开 `diffusion-models.ipynb`，内核选 **Python (computer-vision)**。

**关键结论**：DDPM 训练是显式 MSE（预测噪声），天然稳定；采样 T 步迭代是速度瓶颈；真实实现用 U-Net + VAE 潜空间 + 文本条件（下一课）。
