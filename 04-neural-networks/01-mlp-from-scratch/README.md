# MLP from Scratch

用 numpy 从零实现多层感知机：前向、反向传播（BCE + sigmoid）、训练循环、决策边界，并与 torch autograd 梯度对拍。

**运行**：VS Code 打开 `mlp-from-scratch.ipynb`，内核选 **Python (neural-networks)**。

**前置**：强烈建议先完成 [calculus 05](../02-calculus/05-autograd-backpropagation)

**关键结论**：梯度对拍是全课核心——推导对错一测便知；以后用框架遇到梯度 bug，回到手写推导排查。
