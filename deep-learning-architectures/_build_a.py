# -*- coding: utf-8 -*-
"""Build deep-learning-architectures lessons 01-03."""
from _common import md, code, IMPORTS, write_nb, write_readme

# ============================================================
# Lesson 01: CNN Basics
# ============================================================
c = []
c.append(md("""# CNN Basics

卷积神经网络用"局部、共享权重"的运算替代全连接。本课从零实现卷积、理解 im2col 的矩阵视角、池化与感受野。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 从全连接到卷积
"""))
c.append(md("""全连接层把每个输出与**所有**输入相连：$y_j = \\sum_i w_{ji}x_i$。图像有 3 个问题：

1. **参数爆炸**：$256\\times256\\times3$ 输入 → 全连接层就有几十万参数
2. **忽略局部结构**：相邻像素强相关，远处像素无关
3. **不平移不变**：猫在图左还是图右，参数完全不同

**卷积**只用一个小核（如 $3\\times3$）滑过图像：**局部连接 + 权重共享**。参数从 $O(WH)$ 降到 $O(k_h k_w)$。
"""))

c.append(md("""## 2. 卷积运算：互相关
"""))
c.append(md("""$$(x * w)[i, j] = \\sum_{a,b} x[i{+}a,\\, j{+}b]\\, w[a, b]$$

严格说神经网络用的是**互相关**（不翻转核），习惯上仍叫卷积。步长 stride、填充 padding 控制输出尺寸：

$$H_{out} = \\left\\lfloor \\frac{H + 2\\cdot\\text{pad} - k_h}{\\text{stride}} \\right\\rfloor + 1$$
"""))
c.append(code("""def conv2d_manual(x, w, stride=1, pad=0):
    H, W = x.shape; kh, kw = w.shape
    xp = np.pad(x, pad)
    Ho = (H + 2*pad - kh)//stride + 1
    Wo = (W + 2*pad - kw)//stride + 1
    out = np.zeros((Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            out[i, j] = np.sum(xp[i*stride:i*stride+kh, j*stride:j*stride+kw] * w)
    return out

# 与 torch 对拍
rng = np.random.default_rng(0)
x = rng.standard_normal((7, 7)); w = rng.standard_normal((3, 3))
out_m = conv2d_manual(x, w, stride=1, pad=1)
out_t = F.conv2d(torch.tensor(x).view(1,1,7,7),
                 torch.tensor(w).view(1,1,3,3), padding=1).squeeze().numpy()
print(f"输出尺寸: {out_m.shape}（7×7 + pad1 - 3 + 1 = 7 ✓）")
print(f"与 torch 最大误差: {np.abs(out_m - out_t).max():.2e}")
"""))
c.append(code("""# 边缘检测：一个 3×3 核即可"找到"竖直边缘
img = np.zeros((16, 16))
img[:, 6:10] = 1.0                       # 竖直亮条
k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])   # Sobel 竖直边缘核

edge = conv2d_manual(img, k)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('输入图像（竖直亮条）')
axes[1].imshow(k, cmap='gray'); axes[1].set_title('3×3 卷积核')
axes[2].imshow(edge, cmap='RdBu'); axes[2].set_title('卷积输出（边缘响应）')
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
"""))

c.append(md("""## 3. im2col：卷积的矩阵视角
"""))
c.append(md("""把每个感受野窗口拉成一行（`im2col`），卷积就变成**矩阵乘法** $\\text{cols} \\times w$——直接复用线性代数与矩阵运算优化：

$$\\text{conv}(x, w) = \\text{im2col}(x) \\cdot w_{\\text{flat}}$$
"""))
c.append(code("""def im2col(x, kh, kw, stride=1, pad=0):
    xp = np.pad(x, pad)
    H, W = xp.shape
    Ho = (H - kh)//stride + 1; Wo = (W - kw)//stride + 1
    cols = np.zeros((Ho*Wo, kh*kw))
    for i in range(Ho):
        for j in range(Wo):
            cols[i*Wo+j] = xp[i*stride:i*stride+kh, j*stride:j*stride+kw].ravel()
    return cols

cols = im2col(x, 3, 3, stride=1, pad=1)
conv_mat = (cols @ w.ravel()).reshape(7, 7)
print(f"im2col 形状: {cols.shape}（49 窗口 × 9 核元素）")
print(f"矩阵乘法卷积 vs 手写卷积 最大误差: {np.abs(conv_mat - out_m).max():.2e}")
print("→ 现代框架（cuDNN 等）正是用 im2col + GEMM 加速卷积")
"""))

c.append(md("""## 4. 池化与感受野
"""))
c.append(md("""**池化**（下采样）：取局部窗口的最大值/平均值，输出尺寸减半。作用：降计算量、提供局部平移不变。

**感受野**：输出像素"看到"的输入区域大小。第 $l$ 层感受野递推（$s_i$ 为到第 $i$ 层的步长累积）：

$$r_l = r_{l-1} + (k_l - 1)\\prod_{i<l} s_i$$
"""))
c.append(code("""def maxpool_manual(x, k=2, stride=2):
    Ho = (x.shape[0] - k)//stride + 1; Wo = (x.shape[1] - k)//stride + 1
    out = np.zeros((Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            out[i, j] = x[i*stride:i*stride+k, j*stride:j*stride+k].max()
    return out

print("最大池化 4×4 →", maxpool_manual(rng.standard_normal((4,4))).shape)

# 感受野表：LeNet 风格堆叠
kernels = [5, 2, 5, 2]          # conv5 → pool2 → conv5 → pool2
strides = [1, 2, 1, 2]
r, cum_s = 1, 1
print("\\n层 | 核 | 步长 | 感受野")
for i, (k, s) in enumerate(zip(kernels, strides)):
    r = r + (k - 1) * cum_s
    print(f" {i+1} | {k} | {s} | {r}")
    cum_s *= s
print(f"\\n最终感受野 = {r}×{r}（原始 32×32 输入下，最后特征图的每个点看到 14×14）")
"""))

c.append(md("""## 5. 现代 CNN 组件
"""))
c.append(md("""- **1×1 卷积**：在通道维做线性组合（点卷积），改变通道数、加非线性，参数极少
- **padding='same'**：保持分辨率不变（$k$ 奇时 pad $=(k-1)/2$）
- **stride>1 替代池化**：现代网络常用 stride=2 卷积下采样（可学习）
- **深度可分离卷积**：先逐通道、再 1×1 混合 → MobileNet 的核心，参数骤降
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **手推尺寸**：$28\\times28$ 输入，$5\\times5$ 核，pad=2，stride=1 → 输出几乘几？
2. **实现**：给 `conv2d_manual` 加 `stride` 支持并验证输出尺寸公式。
3. **多种边缘核**：试 Sobel 水平核、拉普拉斯核，观察输出差异。
4. **im2col 多通道**：把 im2col 推广到 3 通道输入（cols 形状变成什么？）。
5. **感受野**：VGG 的 $3\\times3$ 卷积堆 3 层，感受野是多少？为什么等价一个 $7\\times7$ 但参数更少？
"""))

write_nb("01-cnn-basics", "cnn-basics.ipynb", c)
write_readme("01-cnn-basics", """# 01 · CNN Basics

卷积神经网络基础：卷积运算（手写 vs torch 对拍）、边缘检测、im2col 矩阵视角、池化与感受野递推、现代 CNN 组件。

## 前置要求

- 已按 [`../guide.md`](../guide.md) 配置好虚拟环境
- 建议先完成 linear-algebra 模块（矩阵乘法视角）

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 从全连接到卷积 |
| 2 | 卷积运算：互相关 |
| 3 | im2col：矩阵视角 |
| 4 | 池化与感受野 |
| 5 | 现代 CNN 组件 |
| - | 课后练习 |

## 学习建议

- 感受野递推公式是面试高频题，务必自己推一遍
- "卷积 = 局部线性变换"的矩阵视角，是把 CNN 接回线性代数的桥梁
""")

# ============================================================
# Lesson 02: CNN Classic Networks
# ============================================================
c = []
c.append(md("""# CNN Classic Networks

从 LeNet 到 ResNet：为什么网络越做越深？残差连接如何让"深"成为可能？本课用梯度范数实验直接验证残差的作用，并训练一个小 CNN。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 经典网络演进
"""))
c.append(md("""| 网络 | 年份 | 层数(卷积) | 关键创新 |
|------|------|-----------|----------|
| LeNet-5 | 1998 | 2 | 卷积+池化+全连接范式 |
| AlexNet | 2012 | 5 | ReLU、Dropout、GPU、数据增强 |
| VGG | 2014 | 16-19 | 小核堆叠（3×3×3 ≈ 7×7） |
| GoogLeNet | 2014 | 22 | Inception 多尺度 |
| ResNet | 2015 | 50-152 | 残差连接 → 超深网络 |
| DenseNet | 2017 | 121+ | 密集连接 |

趋势：**更深 + 更宽 + 更高效的连接方式**。
"""))

c.append(md("""## 2. 残差连接：让"深"成为可能
"""))
c.append(md("""普通块学习映射 $F(x)$；残差块学习**残差** $H(x) = x + F(x)$：

$$y = x + F(x, W)$$

- 梯度多一条"恒等捷径"：$\\partial y/\\partial x = 1 + \\partial F/\\partial x$，深层梯度不会消失
- 极端情况下 $F \\to 0$，网络退化为恒等——**深度增加不会变差**
"""))
c.append(code("""def block_grad_norms(use_residual, depth=12, seed=0):
    torch.manual_seed(seed)
    layers = [nn.Linear(16, 16) for _ in range(depth)]
    x = torch.randn(1, 16)
    h = x
    for lin in layers:
        h2 = torch.tanh(lin(h))
        h = h + h2 if use_residual else h2
    loss = h.pow(2).mean()
    norms = []
    for lin in layers:
        g = torch.autograd.grad(loss, lin.weight, retain_graph=True)[0]
        norms.append(g.norm().item())
    return np.array(norms)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, use_r in zip(axes, [False, True]):
    norms = block_grad_norms(use_r)
    ax.plot(norms, 'o-')
    ax.set_yscale('log')
    ax.set_xlabel('层（从输出往输入）'); ax.set_ylabel('|∂L/∂W| 范数')
    ax.set_title('残差连接' if use_r else '普通堆叠')
    ax.grid(alpha=0.3)
plt.tight_layout()
print("→ 普通堆叠：梯度随层数指数衰减；残差连接：梯度保持同一量级")
"""))

c.append(md("""## 3. 残差块的实现
"""))
c.append(code("""class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(x + h)          # 残差捷径

x = torch.randn(2, 8, 16, 16)
y = ResidualBlock(8)(x)
print("输入:", tuple(x.shape), "→ 输出:", tuple(y.shape), "（分辨率与通道不变 ✓）")
"""))

c.append(md("""## 4. 训练一个小 CNN（合成条形图分类）
"""))
c.append(md("""离线生成 8×8 二分类图像数据（横条 vs 竖条），用小 CNN 训练——验证"卷积能学出结构特征"。
"""))
c.append(code("""rng = np.random.default_rng(0)
N = 800
X = np.zeros((N, 1, 8, 8)); y = np.zeros(N, dtype=np.int64)
for i in range(N):
    if rng.random() < 0.5:
        row = rng.integers(0, 8); X[i, 0, row, :] = 1.0; y[i] = 0
    else:
        col = rng.integers(0, 8); X[i, 0, :, col] = 1.0; y[i] = 1

fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
for i, ax in enumerate(axes):
    ax.imshow(X[i, 0], cmap='gray')
    ax.set_title('横条' if y[i] == 0 else '竖条')
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
"""))
c.append(code("""class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(8*2*2, 2)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        return self.fc(x.flatten(1))

torch.manual_seed(0)
model = TinyCNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y)

losses, accs = [], []
for step in range(400):
    opt.zero_grad()
    idx = torch.randint(0, N, (64,))
    loss = F.cross_entropy(model(Xt[idx]), yt[idx])
    loss.backward(); opt.step()
    if step % 50 == 0:
        with torch.no_grad():
            acc = (model(Xt).argmax(1) == yt).float().mean().item()
        losses.append(loss.item()); accs.append(acc)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(losses, 'o-'); axes[0].set_xlabel('step/50'); axes[0].set_ylabel('loss'); axes[0].grid(alpha=0.3)
axes[1].plot(accs, 'o-'); axes[1].set_xlabel('step/50'); axes[1].set_ylabel('准确率'); axes[1].grid(alpha=0.3)
axes[0].set_title('训练损失'); axes[1].set_title('训练准确率')
print(f"最终准确率 = {accs[-1]:.3f}")
"""))

c.append(md("""## 5. 从 8×8 到真实图像
"""))
c.append(md("""真实任务（ImageNet 224×224、CIFAR 32×32）只是把这里的 8×8 换成真实数据集（`torchvision.datasets`），结构不变：

$$\\text{conv}^{\\times N} \\to \\text{pool} \\to \\text{conv}^{\\times M} \\to \\text{pool} \\to \\text{flatten} \\to \\text{FC} \\to \\text{softmax}$$

现代实践用 `torchvision.models.resnet18(pretrained=True)` 直接迁移学习，但**手搭一遍**才能理解每个组件的作用。
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **改数据**：把条形图改成"左上亮 vs 右下亮"（对角线模式），CNN 还能学吗？
2. **加深**：堆 4 个残差块训练，对比收敛速度与准确率。
3. **去掉 BN**：把 BatchNorm 去掉重训，观察收敛变化（衔接 05 课初始化）。
4. **VGG 对比**：用 3 个 3×3 卷积（无残差）与 1 个 7×7 卷积比较参数数量。
5. **思考**：残差连接为什么对"梯度消失"的缓解作用那么强？（提示：恒等捷径的导数）
"""))

write_nb("02-cnn-classic-networks", "cnn-classic-networks.ipynb", c)
write_readme("02-cnn-classic-networks", """# 02 · CNN Classic Networks

经典网络：LeNet→AlexNet→VGG→ResNet 演进、残差连接的梯度实验验证、残差块实现、小 CNN 训练（合成条形图）。

## 前置要求

- 先完成 [01-cnn-basics](../01-cnn-basics)
- calculus 08（梯度消失）与 nn-core 05（初始化）对照

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 经典网络演进 |
| 2 | 残差连接：梯度实验 |
| 3 | 残差块实现 |
| 4 | 小 CNN 训练 |
| 5 | 从 8×8 到真实图像 |
| - | 课后练习 |

## 学习建议

- 残差梯度的 log 图是本课最值得反复看的图：一句话解释"为什么深了还能训"
- 合成数据的价值：可控、快速、聚焦结构学习
""")

# ============================================================
# Lesson 03: Attention Mechanism
# ============================================================
c = []
c.append(md("""# Attention Mechanism

注意力 = "让模型自己决定看哪里"。本课从检索类比出发，实现缩放点积注意力，解释为什么除以 $\\sqrt{d_k}$，并实现多头注意力。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 从检索到注意力
"""))
c.append(md("""数据库检索：用 query 匹配 key，取出对应的 value。注意力把它变成**可微、可学习**的版本：

- **Query** $Q$：我要找什么
- **Key** $K$：每个位置提供什么标签
- **Value** $V$：每个位置的内容

匹配强度（相似度）→ softmax 归一化成权重 → 加权求和 value。

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\Big(\\frac{QK^\\mathsf{T}}{\\sqrt{d_k}}\\Big)V$$
"""))
c.append(code("""B, T, d = 2, 5, 8
Q = torch.randn(B, T, d); K = torch.randn(B, T, d); V = torch.randn(B, T, d)

scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
probs = torch.softmax(scores, dim=-1)
out = probs @ V

# 与 PyTorch 官方实现对拍
out_ref = F.scaled_dot_product_attention(Q, K, V)
print(f"输出形状: {out.shape}")
print(f"与 F.scaled_dot_product_attention 最大误差: {torch.abs(out - out_ref).max():.2e}")
print(f"注意力权重行和: {probs[0,0].sum():.4f}（每行是概率分布 ✓）")
"""))

c.append(md("""## 2. 为什么要除以 √d_k：方差分析
"""))
c.append(md("""设 $q_i, k_j$ 各分量独立零均值方差 1（$d$ 维），则点积 $q \\cdot k$ 的方差为 $d$（独立同分布求和）：

$$\\mathrm{Var}[q\\cdot k] = \\sum_{i=1}^d \\mathrm{Var}[q_i k_i] = d$$

不缩放时，$d$ 越大 softmax 输入越极端 → 概率坍缩到 one-hot（梯度消失）。除以 $\\sqrt d$ 把方差拉回 1。
"""))
c.append(code("""rng = np.random.default_rng(0)
for d in [1, 8, 64, 256]:
    Q = rng.standard_normal((2000, d)); K = rng.standard_normal((2000, d))
    dots = (Q * K).sum(1)
    raw = np.exp(dots - dots.max()); p_raw = raw/raw.sum()
    scaled = np.exp(dots/np.sqrt(d) - (dots/np.sqrt(d)).max()); p_s = scaled/scaled.sum()
    print(f"d={d:4d}  点积标准差={dots.std():5.2f}（≈√d）  不缩放 max 概率={p_raw.max():.3f}  缩放后={p_s.max():.3f}")
print("→ 不缩放：d 大时 softmax 趋于 one-hot；除以 √d 后分布保持平滑")
"""))

c.append(md("""## 3. 注意力矩阵的直觉
"""))
c.append(code("""# 一个语义相似度示例：词向量点积 → 注意力权重
words = ['猫', '狗', '鱼', '吃', '游']
emb = torch.randn(5, 16)
sim = emb @ emb.T
probs = torch.softmax(sim / math.sqrt(16), dim=-1)

plt.figure(figsize=(5.5, 4.5))
plt.imshow(probs.numpy(), cmap='Blues')
plt.colorbar(label='注意力权重')
plt.xticks(range(5), words); plt.yticks(range(5), words)
plt.title('注意力矩阵：每行 = 一个 query 对所有 key 的分布')
plt.tight_layout()
"""))

c.append(md("""## 4. 多头注意力
"""))
c.append(md("""单头注意力只能学习一种"匹配模式"。多头把 $d_{model}$ 切成 $n_{heads}$ 个子空间，各自独立注意力再拼接：

$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h)W^O$$

不同头可以分别关注"位置关系""语义相似""句法角色"。
"""))
c.append(code("""class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, _ = x.shape
        Q = self.Wq(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.Wo(out)

mha = MultiHeadAttention(16, 4)
x = torch.randn(2, 7, 16)
print("多头输出形状:", tuple(mha(x).shape), "（B, T, d_model 不变 ✓）")
print(f"参数数量 = {sum(p.numel() for p in mha.parameters())}（3×16×16 映射 + 输出 = 1024）")
"""))

c.append(md("""## 5. 注意力的概率视角
"""))
c.append(md("""softmax 权重是"关于 key 位置的概率分布"（衔接概率 06）：熵低 = 注意力集中（只盯一个位置）；熵高 = 注意力分散。**温度**同样适用（概率 08 课）——低温度让注意力更尖锐。
"""))
c.append(code("""probs = torch.softmax(torch.randn(1, 1, 6) / math.sqrt(8), dim=-1)[0, 0]
H = -(probs * torch.log(probs + 1e-12)).sum()
print(f"注意力权重: {probs.numpy().round(3)}")
print(f"熵 = {H.item():.3f}（对比均匀分布熵 = {np.log(6):.3f}）")
print("→ 熵越小，注意力越集中；训练中模型自动学会集中/分散")
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **推导**：写出 $\\mathrm{Var}[q\\cdot k] = d$ 的完整步骤（用零均值与独立性）。
2. **数值验证**：d=256 时，不缩放 softmax 的熵与缩放后的熵相差多少？
3. **多头 vs 单头**：把 MHA 的 n_heads 改成 1，与单头注意力公式对比是否一致。
4. **因果掩码**：给注意力加下三角掩码（未来位置 -inf），验证输出不看未来。
5. **思考**：为什么要"缩放"而不是直接归一化 Q、K？
"""))

write_nb("03-attention-mechanism", "attention-mechanism.ipynb", c)
write_readme("03-attention-mechanism", """# 03 · Attention Mechanism

注意力机制：QKV 检索类比、缩放点积注意力（手写 vs 官方对拍）、√d_k 的方差分析、注意力矩阵直觉、多头注意力实现。

## 前置要求

- 概率 06（熵、softmax 分布）
- linear-algebra（矩阵乘法）

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 从检索到注意力 |
| 2 | 为什么要除以 √d_k |
| 3 | 注意力矩阵直觉 |
| 4 | 多头注意力 |
| 5 | 注意力的概率视角 |
| - | 课后练习 |

## 学习建议

- "除以 √d 保持方差 1"的推导是面试必考，必须独立完成
- 多头 = 子空间分解，与 PCA 的"多方向"思想一脉相承
""")
