# Training Loop

train / val / test 划分、mini-batch 与 shuffle 的影响、完整训练骨架（zero_grad → backward → step）、常见问题速查。

**运行**：VS Code 打开 `training-loop.ipynb`，内核选 **Python (neural-networks)**。

**前置**：先完成 [01](../01-mlp-from-scratch) 与 [02](../02-activations-losses)

**关键结论**：把 train 函数保存为后续所有实验的基座；"zero_grad 忘写"是新手最常见的 bug（梯度是累加的）。
