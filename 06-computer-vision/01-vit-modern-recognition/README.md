# 01 ViT & Modern Recognition

Vision Transformer 与现代识别：把图像切成 patch 当"词"建模，对比 CNN 归纳偏置，动手训练微型 ViT 完成 4 类图像分类。

**运行**：VS Code 打开 `vit-modern-recognition.ipynb`，内核选 **Python (computer-vision)**。

**关键结论**：patch embedding + 位置编码 + [CLS] token；CNN 靠归纳偏置小数据更快，ViT 靠全局建模大数据反超。
