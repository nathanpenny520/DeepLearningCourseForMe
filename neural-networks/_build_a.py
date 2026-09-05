# -*- coding: utf-8 -*-
"""Build neural-networks lessons 01-04."""
from _common import md, code, IMPORTS, write_nb, write_readme

# ============================================================
# Lesson 01: MLP from Scratch
# ============================================================
c = []
c.append(md("""# MLP from Scratch

用 numpy 从零实现一个多层感知机（含前向、反向、训练），再与 torch autograd 对拍梯度。本课把 calculus 05 的反向传播推导变成可以跑起来的代码。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 数据集：两个高斯簇
"""))
c.append(md("""造一个可复现的二维二分类数据集（两个高斯簇），后续课程反复使用。真实项目里替换成 `torchvision.datasets` 等即可。
"""))
c.append(code("""rng = np.random.default_rng(42)
n = 200
X0 = rng.standard_normal((n, 2)) + np.array([-2.0, 0.0])
X1 = rng.standard_normal((n, 2)) + np.array([2.0, 0.0])
X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(n), np.ones(n)])

plt.figure(figsize=(6, 4.5))
plt.scatter(X[y==0, 0], X[y==0, 1], s=8, alpha=0.6, label='类别 0')
plt.scatter(X[y==1, 0], X[y==1, 1], s=8, alpha=0.6, label='类别 1')
plt.xlabel('x₁'); plt.ylabel('x₂'); plt.legend(); plt.grid(alpha=0.3)
plt.title('二维二分类数据集（高斯簇）')
"""))

c.append(md("""## 2. 网络结构：2-4-1
"""))
c.append(md("""结构：输入 2 维 → 隐藏层 4 神经元（$\\tanh$）→ 输出 1 维（sigmoid）。

$$\n\\begin{aligned}\nz_1 &= W_1 x + b_1, & a_1 &= \\tanh(z_1) \\\\\nz_2 &= W_2 a_1 + b_2, & \\hat y &= \\sigma(z_2)\n\\end{aligned}\n$$"""))
c.append(md("""## 3. 反向传播（BCE + sigmoid 的简化）
"""))
c.append(md("""损失：$L = -\\frac{1}{m}\\sum\\big[y\\log\\hat y + (1-y)\\log(1-\\hat y)\\big]$。

关键梯度（calculus 08 已验证 $\\partial L/\\partial z = \\hat y - y$）：

$$\n\\begin{aligned}\n\\frac{\\partial L}{\\partial z_2} &= \\hat y - y \\\\\n\\frac{\\partial L}{\\partial W_2} &= a_1^\\mathsf{T}\\,\\frac{\\partial L}{\\partial z_2}, &\n\\frac{\\partial L}{\\partial z_1} &= \\Big(\\frac{\\partial L}{\\partial z_2}W_2^\\mathsf{T}\\Big)\\odot(1-a_1^2) \\\\\n\\frac{\\partial L}{\\partial W_1} &= x^\\mathsf{T}\\,\\frac{\\partial L}{\\partial z_1}\n\\end{aligned}\n$$"""))
c.append(code("""def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class MLP:
    def __init__(self, dims, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.standard_normal((dims[0], dims[1])) * 0.5
        self.b1 = np.zeros(dims[1])
        self.W2 = r.standard_normal((dims[1], dims[2])) * 0.5
        self.b2 = np.zeros(dims[2])

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, x, y):
        m = len(y)
        dz2 = (self.a2 - y.reshape(-1, 1)) / m        # ∂L/∂z2 = (ŷ−y)/m
        self.dW2 = self.a1.T @ dz2
        self.db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - self.a1**2)                  # tanh 导数
        self.dW1 = x.T @ dz1
        self.db1 = dz1.sum(axis=0)

    def step(self, lr):
        self.W1 -= lr * self.dW1; self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2; self.b2 -= lr * self.db2

    def loss(self, x, y):
        a = self.forward(x)
        eps = 1e-12
        return -np.mean(y*np.log(a+eps) + (1-y)*np.log(1-a+eps))
"""))

c.append(md("""## 4. 手写梯度 vs torch autograd（对拍）
"""))
c.append(md("""在**完全相同的权重**下，手写 backward 与 torch autograd 应该给出相同的梯度——这是检验推导正确性的黄金标准。
"""))
c.append(code("""mlp = MLP([2, 4, 1], seed=0)

# 构造权重完全相同的 torch 模型
tnet = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
with torch.no_grad():
    tnet[0].weight.copy_(torch.tensor(mlp.W1.T)); tnet[0].bias.copy_(torch.tensor(mlp.b1))
    tnet[2].weight.copy_(torch.tensor(mlp.W2.T)); tnet[2].bias.copy_(torch.tensor(mlp.b2))

xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# torch 侧
loss_t = F.binary_cross_entropy(torch.sigmoid(tnet(xt)), yt)
loss_t.backward()
gW1_t = tnet[0].weight.grad.T.numpy(); gW2_t = tnet[2].weight.grad.T.numpy()
gb1_t = tnet[0].bias.grad.numpy();     gb2_t = tnet[2].bias.grad.numpy()

# 手写侧
mlp.forward(X); mlp.backward(X, y)

print(f"loss 手写 = {mlp.loss(X, y):.8f}  vs  torch = {loss_t.item():.8f}")
print(f"dW1 最大误差 = {np.abs(mlp.dW1 - gW1_t).max():.2e}")
print(f"dW2 最大误差 = {np.abs(mlp.dW2 - gW2_t).max():.2e}")
print(f"db1 最大误差 = {np.abs(mlp.db1 - gb1_t).max():.2e}")
print(f"db2 最大误差 = {np.abs(mlp.db2 - gb2_t).max():.2e}")
"""))

c.append(md("""## 5. 训练循环
"""))
c.append(code("""mlp = MLP([2, 4, 1], seed=0)
lr = 0.5
losses = []
for epoch in range(1000):
    mlp.forward(X)
    mlp.backward(X, y)
    mlp.step(lr)
    if epoch % 50 == 0:
        losses.append((epoch, mlp.loss(X, y)))

acc = ((mlp.forward(X).ravel() > 0.5) == y).mean()
print(f"训练完成，准确率 = {acc:.3f}")

plt.figure(figsize=(8, 4))
ep, ls = zip(*losses)
plt.plot(ep, ls, 'o-')
plt.xlabel('epoch'); plt.ylabel('BCE loss')
plt.title('手写 MLP 训练曲线（梯度下降，无 torch autograd）')
plt.grid(alpha=0.3)
"""))

c.append(md("""## 6. 决策边界可视化
"""))
c.append(code("""gx = np.linspace(-4.5, 4.5, 200); gy = np.linspace(-2.5, 2.5, 200)
GX, GY = np.meshgrid(gx, gy)
grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
pred = mlp.forward(grid).reshape(GX.shape)

plt.figure(figsize=(6.5, 5))
plt.contourf(GX, GY, pred, levels=20, cmap='coolwarm', alpha=0.7)
plt.colorbar(label='P(类别 1)')
plt.scatter(X[y==0, 0], X[y==0, 1], s=8, c='navy', alpha=0.7)
plt.scatter(X[y==1, 0], X[y==1, 1], s=8, c='darkred', alpha=0.7)
plt.xlabel('x₁'); plt.ylabel('x₂'); plt.title('手写 MLP 的决策面（概率热图）')
"""))

c.append(md("""## 7. 从手写到框架
"""))
c.append(md("""手工写一遍的价值：看清每个梯度从哪来。但工程上我们直接 `loss.backward()`——`nn.Linear` 封装了权重/偏置、`F.binary_cross_entropy` 封装了损失、优化器封装了参数更新。后续课程都在"框架"层工作，遇到 bug 时回到本课的手写推导排查。
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **改网络**：把隐藏层改成 8 个神经元，重新训练，对比准确率与决策面。
2. **改激活**：把 $\\tanh$ 换成 ReLU（注意 $dz_1$ 公式要改），重新对拍梯度。
3. **改损失**：把 sigmoid+BCE 换成"线性输出 + MSE"，推导并修改 backward。
4. **数值梯度**：用中心差分（calculus 02 的方法）验证本课任意一层权重梯度。
5. **思考**：为什么 $\\partial L/\\partial z_2 = \\hat y - y$ 是"预测减真值"这么简单的形式？
"""))

write_nb("01-mlp-from-scratch", "mlp-from-scratch.ipynb", c)
write_readme("01-mlp-from-scratch", """# 01 · MLP from Scratch

用 numpy 从零实现多层感知机：前向、反向传播（BCE+sigmoid）、训练循环、决策边界，并与 torch autograd 梯度对拍。

## 前置要求

- 已按 [`../guide.md`](../guide.md) 配置好虚拟环境
- 强烈建议先完成 calculus 05（自动微分与反向传播）

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 二维高斯簇数据集 |
| 2 | 网络结构 2-4-1 |
| 3 | 反向传播公式 |
| 4 | 手写梯度 vs torch 对拍 |
| 5 | 训练循环 |
| 6 | 决策边界可视化 |
| 7 | 从手写到框架 |
| - | 课后练习 |

## 学习建议

- 第 4 节的梯度对拍是全课核心：推导对错一测便知
- 以后用框架遇到梯度 bug，回到手写推导排查
""")

# ============================================================
# Lesson 02: Activations & Losses
# ============================================================
c = []
c.append(md("""# Activations & Losses

激活函数给网络"非线性"能力，损失函数定义"好"与"差"。本课对比主流激活与损失，并证明一个关键事实：**没有非线性的多层网络等于一层**。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 激活函数全家桶
"""))
c.append(md("""| 激活 | 公式 | 导数 | 特点 |
|------|------|------|------|
| sigmoid | $\\sigma(z)=\\frac{1}{1+e^{-z}}$ | $\\sigma(1-\\sigma)$ | 输出 (0,1)，两端饱和 |
| tanh | $\\tanh z$ | $1-\\tanh^2 z$ | 输出 (-1,1)，零中心 |
| ReLU | $\\max(0,z)$ | $\\mathbb{1}[z>0]$ | 稀疏、无饱和，负区死 |
| LeakyReLU | $\\max(\\alpha z, z)$ | $\\alpha$ 或 1 | 解决 ReLU 死亡 |
| ELU | $z\\ (z>0),\\ \\alpha(e^z-1)$ | 分段 | 负区平滑 |
"""))
c.append(code("""def act_fns():
    return {
        'sigmoid': (lambda z: 1/(1+np.exp(-z)),
                    lambda z: (1/(1+np.exp(-z)))*(1-1/(1+np.exp(-z)))),
        'tanh':    (lambda z: np.tanh(z), lambda z: 1-np.tanh(z)**2),
        'ReLU':    (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)),
        'LeakyReLU': (lambda z: np.where(z > 0, z, 0.1*z), lambda z: np.where(z > 0, 1.0, 0.1)),
        'ELU':     (lambda z: np.where(z > 0, z, np.exp(z)-1), lambda z: np.where(z > 0, 1.0, np.exp(z))),
    }
z = np.linspace(-4, 4, 300)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, (f, df) in act_fns().items():
    axes[0].plot(z, f(z), label=name)
    axes[1].plot(z, df(z), label=name)
axes[0].set_title('激活函数'); axes[1].set_title('导数（梯度可流动区域）')
for ax in axes: ax.set_xlabel('z'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
"""))

c.append(md("""## 2. 为什么需要非线性：线性堆叠 = 单层
"""))
c.append(md("""两层线性层 $W_2(W_1 x + b_1) + b_2 = (W_1W_2)x + (b_1W_2 + b_2)$——还是一个线性函数！

**证明**：$n$ 层线性层总可合并为单层，表达能力与单层相同。非线性激活（ReLU/tanh…）是"深度"能工作的必要条件。
"""))
c.append(code("""rng = np.random.default_rng(0)
W1 = rng.standard_normal((3, 4)); b1 = rng.standard_normal(4)
W2 = rng.standard_normal((4, 2)); b2 = rng.standard_normal(2)
x = rng.standard_normal(3)

two_layer = (x @ W1 + b1) @ W2 + b2
W_eff = W1 @ W2;  b_eff = b1 @ W2 + b2
one_layer = x @ W_eff + b_eff

print(f"两层线性与单层最大差 = {np.abs(two_layer - one_layer).max():.2e}")
print("→ 数学上完全等价：没有非线性就没有深度")
"""))

c.append(md("""## 3. 损失函数：回归与分类
"""))
c.append(md("""| 场景 | 损失 | 公式 | 分布假设 |
|------|------|------|----------|
| 回归 | MSE | $(y-\\hat y)^2$ | 高斯噪声 |
| 回归 | MAE | $|y-\\hat y|$ | 拉普拉斯噪声 |
| 回归 | Huber | 分段二次/线性 | 抗离群点 |
| 二分类 | BCE | $-y\\log\\hat y -(1-y)\\log(1-\\hat y)$ | 伯努利 |
| 多分类 | CE | $-\\log \\hat p_y$ | 类别分布 |

（对应概率 05 课"损失 = 负对数似然"对照表。）
"""))
c.append(code("""def mse(d): return d**2
def mae(d): return np.abs(d)
def huber(d, delta=1.0):
    a = np.abs(d)
    return np.where(a <= delta, 0.5*a**2, delta*(a - 0.5*delta))

d = np.linspace(-4, 4, 400)
plt.figure(figsize=(8, 4.5))
plt.plot(d, mse(d), label='MSE')
plt.plot(d, mae(d), label='MAE')
plt.plot(d, huber(d), label='Huber (δ=1)')
plt.xlabel('预测误差 y − ŷ'); plt.ylabel('损失')
plt.title('回归损失对比：离群点敏感性')
plt.legend(); plt.grid(alpha=0.3)
print("误差=3 时: MSE =", mse(3), " MAE =", mae(3), " Huber =", huber(3))
print("→ MSE 对离群点惩罚是平方级，MAE 线性，Huber 折中")
"""))

c.append(md("""## 4. 损失的可微性：为什么 0-1 错误率不能直接优化
"""))
c.append(md("""分类的终极指标是 0-1 错误率（$\\mathbb{1}[\\hat y \\ne y]$），但它处处不可微、梯度为零——没法梯度下降。交叉熵是它的"可微替身"：形状接近 0-1 损失，但有平滑的梯度信号。
"""))
c.append(code("""# 二分类：交叉熵 vs 0-1 损失的代理对比
z = np.linspace(-6, 6, 300)                       # logit，y=1
ce = np.log(1 + np.exp(-z))                       # BCE(σ(z), 1)
zero_one = (z < 0).astype(float)

plt.figure(figsize=(8, 4.5))
plt.plot(z, zero_one, 'k--', label='0-1 错误率（不可微）')
plt.plot(z, ce, 'r-', label='BCE（平滑代理）')
plt.xlabel('logit z（y=1）'); plt.ylabel('损失')
plt.title('交叉熵是 0-1 损失的平滑代理')
plt.legend(); plt.grid(alpha=0.3)
print("BCE 在 z→-∞ 时 →", ce[-1], "，0-1 损失 → 1；但 BCE 梯度处处存在")
"""))

c.append(md("""## 5. 选择指南
"""))
c.append(md("""- **回归**：噪声轻 → MSE；有离群点 → Huber；对尺度不敏感 → MAE
- **分类**：输出层用 softmax（多类）或 sigmoid（二类），损失用 CE/BCE——**不要**用 MSE 配分类输出（梯度弱且非凸性差）
- **数值稳定**：`F.cross_entropy` 把 softmax 与 log 合并计算（log-sum-exp 技巧，见 calculus 08），手写时勿分开
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **证明**：$n$ 层线性网络的输出仍是输入的线性函数（数学归纳法）。
2. **画图**：把 LeakyReLU 的 $\\alpha$ 从 0.01 调到 0.5，观察导数形状变化。
3. **数值验证**：用中心差分验证 BCE 对 logit 的导数 = $\\sigma(z)-y$。
4. **对比实验**：同一模型分别用 MSE 与 BCE 训练二分类，比较收敛速度。
5. **思考**：为什么 softmax + MSE 训练分类会"梯度太小"？（提示：饱和区的导数）
"""))

write_nb("02-activations-losses", "activations-losses.ipynb", c)
write_readme("02-activations-losses", """# 02 · Activations & Losses

激活函数与损失函数：五类激活及其导数、线性堆叠=单层的证明、回归/分类损失对照、0-1 损失的平滑代理视角。

## 前置要求

- 先完成 [01-mlp-from-scratch](../01-mlp-from-scratch)
- 概率 05（损失=NLL）对照

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 激活函数全家桶 |
| 2 | 非线性必要性证明 |
| 3 | 损失函数：回归与分类 |
| 4 | 损失的可微性 |
| 5 | 选择指南 |
| - | 课后练习 |

## 学习建议

- "线性堆叠=单层"的证明是理解深度学习的基石
- 输出层与损失要配套（softmax→CE / sigmoid→BCE），不要混搭
""")

# ============================================================
# Lesson 03: Training Loop
# ============================================================
c = []
c.append(md("""# Training Loop

训练循环是深度学习的"日常操作"：数据划分、mini-batch、shuffle、epoch、验证。本课把完整训练流程拆解成规范骨架。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 数据划分：train / val / test
"""))
c.append(md("""- **train**：更新参数
- **validation**：选超参数、监控过拟合（训练时反复查看，会"间接拟合"它）
- **test**：只评估一次，最终汇报泛化能力

常见比例 8:1:1。**严禁**用 test 调参——那会让 test 变成 val 的延长。
"""))
c.append(code("""rng = np.random.default_rng(42)
n = 400
X0 = rng.standard_normal((n, 2)) + np.array([-2.0, 0.0])
X1 = rng.standard_normal((n, 2)) + np.array([2.0, 0.0])
X = np.vstack([X0, X1]); y = np.concatenate([np.zeros(n), np.ones(n)])

idx = rng.permutation(len(X))
n_tr, n_va = int(0.7*len(X)), int(0.15*len(X))
tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
print(f"train {len(tr)} / val {len(va)} / test {len(te)}")
"""))

c.append(md("""## 2. mini-batch 与 shuffle
"""))
c.append(md("""- **batch**：一次梯度更新用的样本数。全量梯度（batch=全部）稳定但慢、易困在平坦区；单样本 SGD 噪声大；mini-batch 折中
- **shuffle**：打乱样本顺序，让每个 batch 的梯度估计接近"真实期望梯度"。**不 shuffle 时**，按类别排序的数据会让每个 batch 只含一类，梯度来回振荡
"""))
c.append(code("""def make_batches(X, y, bs, shuffle=True, seed=0):
    r = np.random.default_rng(seed)
    idx = np.arange(len(X))
    if shuffle:
        r.shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i:i+bs]
        yield X[sel], y[sel]

def train_with_shuffle(X, y, shuffle, seed=0):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    losses = []
    for epoch in range(60):
        ep_loss = []
        for xb, yb in make_batches(X, y, 32, shuffle=shuffle, seed=seed):
            opt.zero_grad()
            xbt = torch.tensor(xb, dtype=torch.float32); ybt = torch.tensor(yb, dtype=torch.float32).unsqueeze(1)
            loss = F.binary_cross_entropy(torch.sigmoid(model(xbt)), ybt)
            loss.backward(); opt.step()
            ep_loss.append(loss.item())
        losses.append(np.mean(ep_loss))
    return losses

l_shuf = train_with_shuffle(X, y, True)
l_noshuf = train_with_shuffle(X, y, False)

plt.figure(figsize=(8, 4.5))
plt.plot(l_shuf, label='shuffle=True')
plt.plot(l_noshuf, label='shuffle=False')
plt.xlabel('epoch'); plt.ylabel('平均 batch 损失')
plt.title('shuffle 对收敛的影响（数据按类别顺序排列时尤其明显）')
plt.legend(); plt.grid(alpha=0.3)
"""))

c.append(md("""## 3. 完整训练骨架
"""))
c.append(md("""标准流程：每个 epoch 内，遍历 batches → 前向 → 算损失 → `zero_grad` → `backward` → `step`；epoch 末在 val 上评估。`zero_grad` 必不可少——梯度默认**累加**。
"""))
c.append(code("""def train(model, opt, Xtr, ytr, Xva, yva, epochs=100, bs=32, seed=0):
    tr_loss, va_loss, va_acc = [], [], []
    xtr = torch.tensor(Xtr, dtype=torch.float32); ytr = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    xva = torch.tensor(Xva, dtype=torch.float32); yva = torch.tensor(yva, dtype=torch.float32).unsqueeze(1)
    for epoch in range(epochs):
        model.train()
        ep = []
        for xb, yb in make_batches(Xtr, ytr, bs, seed=seed+epoch):
            opt.zero_grad()
            loss = F.binary_cross_entropy(torch.sigmoid(model(xb)), yb)
            loss.backward(); opt.step()
            ep.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy(torch.sigmoid(model(xva)), yva).item()
            acc = ((torch.sigmoid(model(xva)).ravel() > 0.5) == yva.ravel()).float().mean().item()
        tr_loss.append(np.mean(ep)); va_loss.append(vl); va_acc.append(acc)
    return tr_loss, va_loss, va_acc

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.1)
tr_l, va_l, va_a = train(model, opt, X[tr], y[tr], X[va], y[va])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(tr_l, label='train'); axes[0].plot(va_l, label='val')
axes[0].set_xlabel('epoch'); axes[0].set_ylabel('BCE loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(va_a); axes[1].set_xlabel('epoch'); axes[1].set_ylabel('val 准确率'); axes[1].grid(alpha=0.3)
axes[0].set_title('训练/验证损失'); axes[1].set_title('验证准确率')
plt.tight_layout()
"""))

c.append(md("""## 4. 训练中的常见问题速查
"""))
c.append(md("""| 现象 | 常见原因 | 检查方法 |
|------|----------|----------|
| loss 为 NaN | 学习率过大 / log 里出现 0 | 打印梯度范数、减小 lr |
| loss 不下降 | 学习率太小 / 数据未归一化 | 曲线放大看、调 lr |
| train 好 val 差 | 过拟合 | 看 train/val 差距（07 课） |
| 振荡剧烈 | batch 太小 / lr 大 | 增大 batch、加动量 |
| 梯度不更新 | zero_grad 缺失 / requires_grad=False | 检查 `w.grad` 是否 None |
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **改 batch**：把 batch size 从 32 改成 8 与 128，比较损失曲线的噪声与收敛速度。
2. **改 lr**：在 [0.01, 0.05, 0.5, 2.0] 中对比，找出梯度下降的"甜点区间"。
3. **实现 Early Stop**：val 损失连续 N 轮不降就停止，返回最优权重。
4. **数据归一化**：把 X 缩放 100 倍后训练，观察损失曲线变化（呼应 05 课初始化）。
5. **思考**：为什么测试集只能看一次？如果反复用 test 调参会发生什么？
"""))

write_nb("03-training-loop", "training-loop.ipynb", c)
write_readme("03-training-loop", """# 03 · Training Loop

训练循环规范：train/val/test 划分、mini-batch 与 shuffle 的影响、完整训练骨架（zero_grad→backward→step）、常见问题速查。

## 前置要求

- 先完成 [01-mlp-from-scratch](../01-mlp-from-scratch) 与 [02-activations-losses](../02-activations-losses)

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 数据划分 |
| 2 | mini-batch 与 shuffle |
| 3 | 完整训练骨架 |
| 4 | 常见问题速查 |
| - | 课后练习 |

## 学习建议

- 把第 3 节的 `train` 函数保存下来，后续所有实验都基于它扩展
- "zero_grad 忘写"是新手最常见的 bug，记住梯度是累加的
""")

# ============================================================
# Lesson 04: Regularization
# ============================================================
c = []
c.append(md("""# Regularization

模型太复杂会记住噪声（过拟合）。正则化 = 给"复杂度"付代价。本课用多项式回归演示 L1/L2，再实现 Dropout 与早停。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 过拟合演示：高次多项式
"""))
c.append(md("""真实函数 $y = \\sin x$，数据含噪声。次数 15 的多项式能穿过所有训练点，但在点之间剧烈振荡——典型的过拟合。
"""))
c.append(code("""rng = np.random.default_rng(0)
x = np.linspace(-3, 3, 40)
y = np.sin(x) + 0.15*rng.standard_normal(40)
xs = np.linspace(-3, 3, 300)

# 封闭解岭回归：(XᵀX + λI)⁻¹Xᵀy
def ridge_poly(x, y, deg, lam, grid):
    V = np.vander(x, deg+1, increasing=True)
    A = V.T @ V + lam*np.eye(deg+1)
    w = np.linalg.solve(A, V.T @ y)
    return np.vander(grid, deg+1, increasing=True) @ w

plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=10, alpha=0.6, label='数据')
plt.plot(xs, np.sin(xs), 'k-', lw=2, label='真实 sin(x)')
plt.plot(xs, ridge_poly(x, y, 15, 0.0, xs), 'r--', lw=1.5, label='次数15 无正则（过拟合）')
plt.plot(xs, ridge_poly(x, y, 15, 1e-2, xs), 'b-', lw=1.5, label='次数15 + L2 λ=0.01')
plt.xlabel('x'); plt.ylabel('y')
plt.title('过拟合 vs L2 正则')
plt.legend(); plt.grid(alpha=0.3)
"""))

c.append(md("""## 2. L1 vs L2：稀疏 vs 收缩
"""))
c.append(md("""- **L2（权重衰减）**：$\\lambda\\|w\\|_2^2$，把大权重按比例**收缩**，不置零
- **L1**：$\\lambda\\|w\\|_1$，鼓励**稀疏**（很多权重恰好为 0）——特征选择
"""))
c.append(code("""V = np.vander(x, 16, increasing=True)
Vn = V / V.std(axis=0)                    # 标准化特征（数值稳定）
Vt = torch.tensor(Vn, dtype=torch.float64); yt = torch.tensor(y, dtype=torch.float64)

# L2 封闭解
w_l2 = torch.linalg.solve(Vt.T@Vt + 0.01*torch.eye(16, dtype=torch.float64), Vt.T@yt)

# L1：梯度下降（近端/次梯度简化版）
w_l1 = torch.zeros(16, dtype=torch.float64, requires_grad=True)
opt = torch.optim.SGD([w_l1], lr=1e-3)
for _ in range(4000):
    opt.zero_grad()
    loss = ((Vt@w_l1 - yt)**2).mean() + 0.05*w_l1.abs().sum()
    loss.backward(); opt.step()

print(f"L1 非零系数数: {(w_l1.abs().detach().numpy() > 1e-3).sum()} / 16")
print(f"L2 非零系数数: {(w_l2.abs().numpy() > 1e-3).sum()} / 16")
print("→ L1 把多数系数压到 0，L2 只收缩不置零")
"""))

c.append(md("""## 3. Dropout：训练时随机失活
"""))
c.append(md("""训练时以概率 $p$ 把神经元置 0（并除以 $1-p$ 保持期望不变），等价于训练"子网络的集成"；推理时全部保留。

$$\\mathbb{E}[\\text{Dropout}(x)] = x \\quad(\\text{因为 } x\\cdot(1-p)/(1-p))$$
"""))
c.append(code("""# 手写 Dropout 验证期望
rng = np.random.default_rng(1)
x = np.array([1.0, 2.0, 3.0, 4.0])
p = 0.5
sums = []
for _ in range(100000):
    mask = (rng.random(4) > p).astype(float) / (1-p)     # 失活后缩放
    sums.append((x*mask).mean())
print(f"E[Dropout(x)] 平均 = {np.mean(sums):.4f}，x 均值 = {x.mean():.4f}")
print("→ 缩放 1/(1-p) 保证期望不变")
"""))
c.append(code("""# Dropout 对泛化的影响：同一模型带/不带 Dropout
def make_batches(X, y, bs, shuffle=True, seed=0):
    r = np.random.default_rng(seed)
    idx = np.arange(len(X))
    if shuffle:
        r.shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i:i+bs]
        yield X[sel], y[sel]

def train(model, opt, Xtr, ytr, Xva, yva, epochs=100, bs=32, seed=0):
    tr_loss, va_loss = [], []
    xtr = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    xva = torch.tensor(Xva, dtype=torch.float32); yva_t = torch.tensor(yva, dtype=torch.float32).unsqueeze(1)
    for epoch in range(epochs):
        model.train()
        ep = []
        for xb, yb in make_batches(Xtr, ytr_t, bs, seed=seed+epoch):
            opt.zero_grad()
            loss = F.binary_cross_entropy(torch.sigmoid(model(xb)), yb)
            loss.backward(); opt.step()
            ep.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy(torch.sigmoid(model(xva)), yva_t).item()
        tr_loss.append(np.mean(ep)); va_loss.append(vl)
    return tr_loss, va_loss

def run(dropout_p, seed=0):
    torch.manual_seed(0)
    layers = [nn.Linear(2, 32), nn.ReLU()]
    if dropout_p:
        layers.append(nn.Dropout(dropout_p))
    layers += [nn.Linear(32, 1)]
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    _, va = train(model, opt, X[tr], y[tr], X[va], y[va], epochs=150)
    return va

# 复用上一课的数据划分（若未定义则重新生成）
try:
    X, y, tr, va, te
except NameError:
    rng = np.random.default_rng(42)
    n = 400
    X0 = rng.standard_normal((n, 2)) + np.array([-2.0, 0.0])
    X1 = rng.standard_normal((n, 2)) + np.array([2.0, 0.0])
    X = np.vstack([X0, X1]); y = np.concatenate([np.zeros(n), np.ones(n)])
    idx = rng.permutation(len(X))
    n_tr = int(0.7*len(X)); n_va = int(0.15*len(X))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

tr0, va0, _ = run(0.0)
tr1, va1, _ = run(0.3)

plt.figure(figsize=(8, 4.5))
plt.plot(va0, label='val 无 Dropout')
plt.plot(va1, label='val Dropout=0.3')
plt.xlabel('epoch'); plt.ylabel('val loss')
plt.title('Dropout 降低验证损失（抑制过拟合）')
plt.legend(); plt.grid(alpha=0.3)
"""))

c.append(md("""## 4. 早停与数据增强
"""))
c.append(md("""- **早停（Early Stopping）**：val 损失不再下降就停止，保存最优权重——"免费的正则化"
- **数据增强**：对输入做保标签变换（图像翻转/旋转/加噪），扩大有效样本量
- 两者都是在"减少过拟合"和"不损失拟合能力"之间找平衡
"""))
c.append(code("""# 早停演示：监控 val，连续 15 轮不降就停
def train_early_stop(X, y, tr, va, patience=15, seed=0):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    xtr = torch.tensor(X[tr], dtype=torch.float32); ytr = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
    xva = torch.tensor(X[va], dtype=torch.float32); yva = torch.tensor(y[va], dtype=torch.float32).unsqueeze(1)
    best, wait, best_state = float('inf'), 0, None
    for epoch in range(300):
        model.train()
        for xb, yb in make_batches(X[tr], y[tr], 32, seed=seed+epoch):
            opt.zero_grad()
            loss = F.binary_cross_entropy(torch.sigmoid(model(xb)), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy(torch.sigmoid(model(xva)), yva).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                model.load_state_dict(best_state)
                return epoch, best
    return epoch, best

stop_ep, best_vl = train_early_stop(X, y, tr, va)
print(f"早停在 epoch {stop_ep} 停止，最优 val loss = {best_vl:.4f}")
print("→ 早停 = 训练过程中自动选『恰好不过拟合』的时刻")
"""))

c.append(md("""## 5. 权重衰减与 AdamW
"""))
c.append(md("""L2 正则在 SGD 里等于"权重衰减"：$w \\leftarrow (1-\\eta\\lambda)w - \\eta\\nabla L$。但 Adam 的自适应学习率会破坏这个等价——**AdamW** 把权重衰减放在动量之外，恢复 L2 语义。工程上：用 AdamW 代替 Adam+L2。
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **λ 扫描**：λ ∈ [0, 1e-4, 1e-2, 1] 重跑岭回归，观察曲线从过拟合到欠拟合的转变。
2. **L1 稀疏性**：把 L1 惩罚从 0.05 调到 0.5，统计非零系数变化。
3. **Dropout 位置**：比较"隐藏层前"与"输入层后"加 Dropout 的效果差异。
4. **早停敏感性**：把 patience 改成 5 与 50，观察停止点与最终性能。
5. **思考**：为什么说早停是"免费"的正则化？它等价于限制了什么？
"""))

write_nb("04-regularization", "regularization.ipynb", c)
write_readme("04-regularization", """# 04 · Regularization

正则化：多项式过拟合演示、L1（稀疏）vs L2（收缩）、Dropout 原理与实现、早停与数据增强、AdamW 与权重衰减的关系。

## 前置要求

- 先完成 [03-training-loop](../03-training-loop)

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 过拟合：高次多项式 |
| 2 | L1 vs L2 |
| 3 | Dropout |
| 4 | 早停与数据增强 |
| 5 | 权重衰减与 AdamW |
| - | 课后练习 |

## 学习建议

- L1/L2 的"稀疏 vs 收缩"差异是面试高频考点，务必理解几何直觉（L1 的角点）
- Dropout 的期望不变缩放 1/(1-p) 要会推导
""")
