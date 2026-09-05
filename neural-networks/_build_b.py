# -*- coding: utf-8 -*-
"""Build neural-networks lessons 05-08."""
from _common import md, code, IMPORTS, write_nb, write_readme

# ============================================================
# Lesson 05: Weight Initialization
# ============================================================
c = []
c.append(md("""# Weight Initialization

初始化决定了训练**能不能开始**。本课证明全零初始化的对称性陷阱，推导 Xavier 与 Kaiming 的方差公式，并可视化"激活方差随层数的传播"。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 全零初始化的对称性陷阱
"""))
c.append(md("""如果所有权重初始为 0，同一层内所有神经元收到相同输入、产生相同输出、得到相同梯度——**永远无法打破对称**，网络退化为单神经元。

**证明**：$a_j = \\sum_i w_{ji}x_i$，$w_{ji}=0 \\Rightarrow a_j=0 \\Rightarrow$ 所有 $j$ 的梯度 $\\partial L/\\partial w_{ji}$ 相同。
"""))
c.append(code("""torch.manual_seed(0)
m = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
with torch.no_grad():
    for p in m.parameters():
        p.zero_()

x = torch.randn(8, 2); y = (x[:, 0] > 0).float().unsqueeze(1)
loss = F.binary_cross_entropy(torch.sigmoid(m(x)), y)
loss.backward()

print("隐藏层权重梯度（行 = 神经元）:")
print(m[0].weight.grad.numpy().round(4))
print("→ 4 行完全相同：4 个神经元永远学成同一个，白白浪费容量")
"""))

c.append(md("""## 2. 方差传播与 Xavier 初始化
"""))
c.append(md("""目标：让激活 $h = Wx$ 的**方差跨层不变**（避免逐层放大→爆炸 或 缩小→消失）。

设 $x$ 各分量独立、零均值、方差 $\\mathrm{Var}[x]$，$W$ 各分量独立零均值方差 $\\mathrm{Var}[W]$：

$$\\mathrm{Var}[h_j] = \\sum_{i=1}^{d_{in}} \\mathrm{Var}[w_{ji} x_i] = d_{in}\\,\\mathrm{Var}[W]\\,\\mathrm{Var}[x]$$

要求 $\\mathrm{Var}[h] = \\mathrm{Var}[x]$ ⇒ $\\mathrm{Var}[W] = \\frac{1}{d_{in}}$（前向视角）。反向传播对称地要求 $\\mathrm{Var}[W] = \\frac{1}{d_{out}}$。**Xavier** 取两者调和：

$$\\mathrm{Var}[W] = \\frac{2}{d_{in} + d_{out}}$$
"""))
c.append(md("""## 3. Kaiming（ReLU 修正）
"""))
c.append(md("""ReLU 会把一半输入置零（负区），实际方差减半：$\\mathrm{Var}[h] = \\frac{d_{in}}{2}\\mathrm{Var}[W]\\mathrm{Var}[x]$。

要求方差守恒 ⇒ $$\\mathrm{Var}[W] = \\frac{2}{d_{in}}$$

tanh/sigmoid 用 Xavier，ReLU 用 Kaiming——**初始化要和激活函数匹配**。
"""))
c.append(code("""def activation_stds(depth, fan_in, init, activation, seed=0):
    r = np.random.default_rng(seed)
    x = r.standard_normal(fan_in)
    stds = [x.std()]
    for _ in range(depth):
        if init == 'plain':
            W = r.standard_normal((fan_in, fan_in)) * 0.01
        elif init == 'big':
            W = r.standard_normal((fan_in, fan_in)) * 1.0
        elif init == 'xavier':
            W = r.standard_normal((fan_in, fan_in)) * np.sqrt(2.0 / (2*fan_in))
        elif init == 'kaiming':
            W = r.standard_normal((fan_in, fan_in)) * np.sqrt(2.0 / fan_in)
        x = W @ x
        x = np.tanh(x) if activation == 'tanh' else np.maximum(0, x)
        stds.append(x.std())
    return np.array(stds)

L = 20; fin = 128
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for init, act, ax, title in [
    ('plain', 'tanh', axes[0], 'tanh：plain(0.01) vs Xavier'),
    ('xavier', 'tanh', axes[0], None),
    ('big', 'tanh', axes[0], None),
    ('plain', 'relu', axes[1], 'ReLU：plain(0.01) vs Kaiming'),
    ('kaiming', 'relu', axes[1], None),
]:
    s = activation_stds(L, fin, init, act)
    axes[0 if act == 'tanh' else 1].semilogy(s, label=init)
axes[0].set_title('tanh：plain(0.01) vs Xavier'); axes[1].set_title('ReLU：plain(0.01) vs Kaiming')
for ax in axes:
    ax.set_xlabel('层'); ax.set_ylabel('激活标准差'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
print("plain: 激活标准差随层数指数衰减（梯度消失）")
print("big  : 激活迅速饱和到 ±1（梯度消失的另一形态）")
print("Xavier/Kaiming: 标准差稳定在 ~1（信号畅通）")
"""))

c.append(md("""## 4. 与梯度消失/爆炸的关联
"""))
c.append(md("""激活方差失控 ⇒ 反向传播梯度也失控（链式法则连乘）：

$$\\frac{\\partial L}{\\partial W^{(1)}} = \\Big(\\frac{\\partial L}{\\partial z^{(k)}} W^{(k)}\\Big)\\cdots\\frac{\\partial a^{(1)}}{\\partial z^{(1)}}\\,x$$

好的初始化把乘积控制在 ~1 附近。ResNet 的残差连接、BatchNorm 都是"让信号穿越深层"的补充手段（架构模块 02 课会验证残差的梯度效果）。
"""))

c.append(md("""## 5. torch 的默认初始化
"""))
c.append(code("""torch.manual_seed(0)
lin = nn.Linear(128, 128)
with torch.no_grad():
    w = lin.weight.numpy()
print(f"nn.Linear 默认: std = {w.std():.4f}，期望 √(2/256) = {np.sqrt(2/256):.4f}")
print("→ PyTorch 默认用 Kaiming 均匀分布，方差与推导一致")
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **推导**：写出 $\\mathrm{Var}[h] = d_{in}\\mathrm{Var}[W]\\mathrm{Var}[x]$ 的完整步骤（用独立性 + 零均值）。
2. **数值验证**：用中心极限检验，$d_{in}=64$ 时 $h$ 是否接近高斯（画直方图）。
3. **实验**：Xavier 配 ReLU、Kaiming 配 tanh 分别训练，观察哪个更差，为什么。
4. **大 fan**：把 fan_in 从 128 改成 512，重跑方差传播，验证公式与层数无关。
5. **思考**：BatchNorm 出现后，初始化还重要吗？为什么？（提示：BN 重置尺度）
"""))

write_nb("05-weight-initialization", "weight-initialization.ipynb", c)
write_readme("05-weight-initialization", """# 05 · Weight Initialization

权重初始化：全零对称性陷阱、方差传播推导、Xavier（2/(d_in+d_out)）、Kaiming（2/d_in）、激活方差随层数传播的可视化。

## 前置要求

- 先完成 [01-mlp-from-scratch](../01-mlp-from-scratch)
- 概率 02（方差）与 calculus 08（梯度消失）对照

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 全零初始化对称性 |
| 2 | 方差传播与 Xavier |
| 3 | Kaiming（ReLU 修正） |
| 4 | 与梯度消失的关联 |
| 5 | torch 默认初始化 |
| - | 课后练习 |

## 学习建议

- 方差传播公式只需记住"Var[h] = d_in·Var[W]·Var[x]"这一条，其余都是推导
- 初始化与激活匹配（Xavier↔tanh、Kaiming↔ReLU）是工程要点
""")

# ============================================================
# Lesson 06: Optimizers in Practice
# ============================================================
c = []
c.append(md("""# Optimizers in Practice

从 SGD 到 Adam：动量、自适应学习率如何加速收敛。本课在同一个任务上横评四种优化器，并演示学习率调度、warmup 与梯度裁剪。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 四种优化器的更新规则
"""))
c.append(md("""| 优化器 | 更新规则 | 关键思想 |
|--------|----------|----------|
| SGD | $w \\gets w - \\eta g$ | 基础 |
| Momentum | $v \\gets \\beta v + g;\\ w \\gets w - \\eta v$ | 沿历史方向累积，穿越山谷 |
| RMSprop | $v \\gets \\beta v + (1-\\beta)g^2;\\ w \\gets w - \\frac{\\eta}{\\sqrt{v}+\\epsilon}g$ | 按梯度幅度逐参数缩放 |
| Adam | 动量 + RMSprop + 偏差校正 | 两者的结合 |

直觉：动量看"方向"，RMSprop/Adam 看"每维的尺度"。
"""))
c.append(code("""rng = np.random.default_rng(42)
n = 400
X0 = rng.standard_normal((n, 2)) + np.array([-2.0, 0.0])
X1 = rng.standard_normal((n, 2)) + np.array([2.0, 0.0])
X = np.vstack([X0, X1]); y = np.concatenate([np.zeros(n), np.ones(n)])
Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

def run_opt(make_opt, steps=400, bs=64, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = make_opt(model.parameters())
    losses = []
    for step in range(steps):
        opt.zero_grad()
        idx = torch.randint(0, len(Xt), (bs,))
        loss = F.binary_cross_entropy(torch.sigmoid(model(Xt[idx])), yt[idx])
        loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

curves = {
    'SGD lr=0.05':        run_opt(lambda p: torch.optim.SGD(p, lr=0.05)),
    'Momentum 0.9':       run_opt(lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    'RMSprop lr=0.005':   run_opt(lambda p: torch.optim.RMSprop(p, lr=0.005)),
    'Adam lr=0.01':       run_opt(lambda p: torch.optim.Adam(p, lr=0.01)),
}
plt.figure(figsize=(9, 5))
for name, ls in curves.items():
    plt.plot(ls, label=name)
plt.xlabel('step'); plt.ylabel('BCE loss')
plt.title('同一网络、同一数据：优化器横评')
plt.legend(); plt.grid(alpha=0.3)
"""))

c.append(md("""## 2. Adam 的超参数
"""))
c.append(md("""- $\\beta_1$（一阶动量，默认 0.9）：方向平滑
- $\\beta_2$（二阶动量，默认 0.999）：尺度估计
- $\\epsilon$（默认 1e-8）：防除零
- 偏差校正：前几步估计偏小，$\\hat v_t = v_t/(1-\\beta_2^t)$ 修正

**实践**：先试默认；loss 不稳就降 lr；$\\epsilon$ 太小（1e-8）在低精度训练可能出 NaN，可用 1e-6~1e-4。
"""))
c.append(code("""# Adam 不同学习率
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for lr in [0.001, 0.01, 0.05]:
    axes[0].plot(run_opt(lambda p: torch.optim.Adam(p, lr=lr), steps=200), label=f'lr={lr}')
axes[0].set_title('Adam：学习率'); axes[0].set_xlabel('step'); axes[0].legend(); axes[0].grid(alpha=0.3)

for eps in [1e-8, 1e-4, 1e-2]:
    axes[1].plot(run_opt(lambda p: torch.optim.Adam(p, lr=0.01, eps=eps), steps=200), label=f'eps={eps}')
axes[1].set_title('Adam：ε'); axes[1].set_xlabel('step'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
"""))

c.append(md("""## 3. 学习率调度
"""))
c.append(md("""训练中逐步降低学习率：先大步探索，后小步精调。

- **StepLR**：每 N 轮 ×γ
- **CosineAnnealing**：余弦从初始到 0，平滑下降
"""))
c.append(code("""def run_schedule(make_sched, steps=400, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = make_sched(opt)
    losses, lrs = [], []
    for step in range(steps):
        opt.zero_grad()
        idx = torch.randint(0, len(Xt), (64,))
        loss = F.binary_cross_entropy(torch.sigmoid(model(Xt[idx])), yt[idx])
        loss.backward(); opt.step(); sched.step()
        losses.append(loss.item()); lrs.append(opt.param_groups[0]['lr'])
    return losses, lrs

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
l_const, _ = run_schedule(lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0))
l_step, lr_step = run_schedule(lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5))
l_cos, lr_cos = run_schedule(lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400))
axes[0].plot(l_const, label='constant'); axes[0].plot(l_step, label='StepLR'); axes[0].plot(l_cos, label='Cosine')
axes[0].set_xlabel('step'); axes[0].set_ylabel('loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_title('损失曲线')
axes[1].plot(lr_step, label='StepLR'); axes[1].plot(lr_cos, label='Cosine')
axes[1].set_xlabel('step'); axes[1].set_ylabel('学习率'); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_title('学习率曲线')
plt.tight_layout()
"""))

c.append(md("""## 4. Warmup 与梯度裁剪
"""))
c.append(md("""- **Warmup**：前若干步学习率从 0 线性升到目标值——避免开局大梯度破坏预训练/大 batch 的统计
- **梯度裁剪**：$g \\gets g \\cdot \\min(1, \\frac{\\text{clip}}{\\|g\\|})$——把梯度范数限制在阈值内，防爆炸（尤其 RNN/Transformer）
"""))
c.append(code("""def run_with_clip(clip, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for step in range(400):
        opt.zero_grad()
        idx = torch.randint(0, len(Xt), (64,))
        loss = F.binary_cross_entropy(torch.sigmoid(model(Xt[idx])), yt[idx])
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(loss.item())
    return losses

plt.figure(figsize=(8, 4.5))
plt.plot(run_with_clip(None), label='无裁剪')
plt.plot(run_with_clip(1.0), label='裁剪到 1.0')
plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.grid(alpha=0.3)
plt.title('梯度裁剪：防爆炸（本任务温和，差异小；RNN/大模型场景关键）')
"""))

c.append(md("""## 5. 实操清单
"""))
c.append(md("""1. **默认起手**：Adam(lr=1e-3)，不行再调
2. **收敛不理想**：试 SGD+Momentum(0.9) + 余弦调度（某些任务泛化更好）
3. **大模型/Transformer**：AdamW + warmup + 梯度裁剪
4. **小数据小模型**：SGD 足够，省内存
5. **遇到 NaN**：降 lr、查 ε、查数据、查 loss 公式
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **动手实现**：不用 torch.optim，手写 Momentum 与 RMSprop 的更新步骤（对照公式）。
2. **超参扫描**：对 Adam 扫 lr ∈ {1e-4, 1e-3, 1e-2, 1e-1}，记录最优值。
3. **余弦 vs Step**：把 StepLR 的 γ 改成 0.9、step 改成 50，比较与余弦的差异。
4. **裁剪阈值**：把 clip 从 1.0 改成 0.1 与 10，观察损失曲线。
5. **思考**：为什么自适应优化器（Adam）在测试集上有时不如调好的 SGD+Momentum？
"""))

write_nb("06-optimizers-in-practice", "optimizers-in-practice.ipynb", c)
write_readme("06-optimizers-in-practice", """# 06 · Optimizers in Practice

优化器实践：SGD/Momentum/RMSprop/Adam 横评、Adam 超参数、学习率调度（StepLR/Cosine）、warmup 与梯度裁剪、实操清单。

## 前置要求

- 先完成 [03-training-loop](../03-training-loop)
- calculus 06（梯度下降理论）对照

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 四种优化器更新规则与横评 |
| 2 | Adam 超参数 |
| 3 | 学习率调度 |
| 4 | Warmup 与梯度裁剪 |
| 5 | 实操清单 |
| - | 课后练习 |

## 学习建议

- 优化器横评要自己跑一遍，感受"收敛速度 vs 泛化"的取舍
- AdamW + warmup + 裁剪是大模型的标配，记住这个组合
""")

# ============================================================
# Lesson 07: Overfitting & Generalization
# ============================================================
c = []
c.append(md("""# Overfitting & Generalization

过拟合不是玄学，而是偏差-方差分解的必然结果。本课推导分解公式，用多项式拟合模拟 bias²/variance 随复杂度的变化，并画出学习曲线。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 偏差-方差分解
"""))
c.append(md("""对固定数据点 $x$，记真值 $y = f(x) + \\epsilon$（噪声 $\\mathrm{Var}[\\epsilon] = \\sigma^2$），模型估计 $\\hat f(x)$（对**数据集**取期望）：

$$\\mathbb{E}\\big[(y - \\hat f)^2\\big] = \\underbrace{\\big(\\mathbb{E}[\\hat f] - f\\big)^2}_{\\text{偏差}^2} + \\underbrace{\\mathrm{Var}[\\hat f]}_{\\text{方差}} + \\underbrace{\\sigma^2}_{\\text{不可约噪声}}$$

**推导**：$y - \\hat f = (f - \\mathbb{E}[\\hat f]) + (\\mathbb{E}[\\hat f] - \\hat f) + \\epsilon$，三项交叉期望为零（$\\epsilon$ 零均值且独立）。

- 简单模型：偏差大（拟合不足）、方差小
- 复杂模型：偏差小、方差大（拟合噪声）
- 总误差在中间某复杂度取最小——这就是"该多复杂"的答案
"""))
c.append(code("""rng = np.random.default_rng(0)
x_grid = np.linspace(-3, 3, 200)
f_true = np.sin(x_grid)
degrees = [1, 3, 6, 9]
bias2, variance = [], []
for d in degrees:
    preds = []
    for _ in range(150):
        xs = rng.uniform(-3, 3, 30)
        ys = np.sin(xs) + 0.2*rng.standard_normal(30)
        coeffs = np.polyfit(xs, ys, d)
        preds.append(np.polyval(coeffs, x_grid))
    preds = np.array(preds)
    bias2.append(((preds.mean(0) - f_true)**2).mean())
    variance.append(preds.var(0).mean())

bias2 = np.array(bias2); variance = np.array(variance)
total = bias2 + variance + 0.2**2
print("次数 | 偏差² | 方差 | 总误差(含噪声)")
for d, b, v, t in zip(degrees, bias2, variance, total):
    print(f"  {d:2d}  | {b:.4f} | {v:.4f} | {t:.4f}")

plt.figure(figsize=(8, 4.8))
plt.plot(degrees, bias2, 'o-', label='偏差²')
plt.plot(degrees, variance, 's-', label='方差')
plt.plot(degrees, total, 'd-', label='总误差 = 偏差²+方差+噪声')
plt.xlabel('多项式次数（模型复杂度）'); plt.ylabel('误差')
plt.title('偏差-方差权衡：总误差在中间复杂度取最小')
plt.legend(); plt.grid(alpha=0.3)
"""))

c.append(md("""## 2. 可视化：三种复杂度下的拟合
"""))
c.append(code("""fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, d in zip(axes, [1, 3, 9]):
    xs = rng.uniform(-3, 3, 25)
    ys = np.sin(xs) + 0.2*rng.standard_normal(25)
    for trial in range(20):                    # 多次重采样数据集
        idx = rng.choice(len(xs), len(xs), replace=True)
        w = np.polyfit(xs[idx], ys[idx], d)
        ax.plot(x_grid, np.polyval(w, x_grid), color='gray', alpha=0.15)
    w = np.polyfit(xs, ys, d)
    ax.plot(x_grid, np.polyval(w, x_grid), 'r-')
    ax.plot(x_grid, f_true, 'k--')
    ax.scatter(xs, ys, s=8)
    ax.set_title(f'次数 {d}'); ax.set_xlim(-3, 3); ax.set_ylim(-2, 2)
plt.tight_layout()
print("灰线 = 不同数据集上的拟合 → 次数 9 的灰线彼此差异巨大（方差大）")
"""))

c.append(md("""## 3. 学习曲线：数据量 vs 泛化差距
"""))
c.append(md("""固定模型复杂度，样本越多：train 误差上升（更难完美记忆）、test 误差下降（规律更清楚），两者**收敛**。若样本很少，train≈0、test 高 → 过拟合。
"""))
c.append(code("""def learning_curve(n_train, deg=5, n_test=2000):
    xs = rng.uniform(-3, 3, n_train)
    ys = np.sin(xs) + 0.2*rng.standard_normal(n_train)
    w = np.polyfit(xs, ys, deg)
    tr_err = np.mean((np.polyval(w, xs) - ys)**2)
    xt = rng.uniform(-3, 3, n_test)
    yt = np.sin(xt) + 0.2*rng.standard_normal(n_test)
    te_err = np.mean((np.polyval(w, xt) - yt)**2)
    return tr_err, te_err

ns = [10, 20, 40, 80, 160, 320]
tr_e = []; te_e = []
for n in ns:
    t1, t2 = learning_curve(n)
    tr_e.append(t1); te_e.append(t2)

plt.figure(figsize=(8, 4.5))
plt.plot(ns, tr_e, 'o-', label='train 误差')
plt.plot(ns, te_e, 's-', label='test 误差')
plt.xscale('log')
plt.xlabel('训练样本数'); plt.ylabel('MSE')
plt.title('学习曲线：样本越多，泛化差距越小')
plt.legend(); plt.grid(alpha=0.3)
"""))

c.append(md("""## 4. 验证集的使用规范
"""))
c.append(md("""- val 是"调参时反复看"的曲线，test 是"交卷前只看一次"的分数
- 每次用 val 选超参数，本质上是在对 val 做轻微的过拟合；超参数越多、试得越久，val 越不可信 → 用**嵌套验证**或增大数据集
- **数据泄漏**：归一化统计量、特征选择必须在 train 上算，再应用到 val/test，否则泄漏会让 val/test 虚高
"""))

c.append(md("""## 5. 实践法则
"""))
c.append(md("""| 症状 | 诊断 | 处方 |
|------|------|------|
| train≈0，val 高 | 过拟合 | 正则化、更多数据、更小模型 |
| train 高，val 高 | 欠拟合 | 更大模型、更长训练、更小正则 |
| train 高，val 更高 | 数据问题/泄漏 | 检查数据划分与预处理 |

先修"欠拟合"（让 train 降下来），再谈"过拟合"（拉近 val）——顺序别反。
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **完整推导**：写出偏差-方差分解的三步（展开、交叉项为零、分别命名）。
2. **噪声项**：把 $\\sigma^2$ 从 0.2 改成 0.05 与 0.8，重跑权衡图，观察最优复杂度的移动方向。
3. **bootstrap**：用自助法（重采样）估计 $\\mathrm{Var}[\\hat f]$，与解析法对比。
4. **学习曲线**：把次数从 5 改成 9，观察 train/test 曲线的起点与收敛值变化。
5. **思考**：为什么说"深度学习过拟合时，增大数据往往比增大正则化更有效"？
"""))

write_nb("07-overfitting-generalization", "overfitting-generalization.ipynb", c)
write_readme("07-overfitting-generalization", """# 07 · Overfitting & Generalization

过拟合与泛化：偏差-方差分解推导、三种复杂度下的拟合可视化、学习曲线、验证集规范与数据泄漏、实践法则。

## 前置要求

- 先完成 [04-regularization](../04-regularization)

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | 偏差-方差分解 |
| 2 | 复杂度可视化 |
| 3 | 学习曲线 |
| 4 | 验证集规范 |
| 5 | 实践法则 |
| - | 课后练习 |

## 学习建议

- 偏差-方差分解是"模型该多复杂"的唯一理论答案，推导务必亲手写
- "先治欠拟合、再治过拟合"的顺序能省掉大量瞎调参
""")

# ============================================================
# Lesson 08: Mini Training Framework
# ============================================================
c = []
c.append(md("""# Mini Training Framework

把前七课的知识组装成一个 100 行内可复用的训练框架：Dataset → DataLoader → Model → Loss → Optimizer → Trainer。换模型/损失/优化器只需改一行。
"""))
c.append(md("""## 0. 环境配置与导入
"""))
c.append(IMPORTS)

c.append(md("""## 1. 数据层：Dataset 与 DataLoader
"""))
c.append(code("""class Dataset:
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class DataLoader:
    def __init__(self, ds, batch_size, shuffle=True, seed=0):
        self.ds, self.bs, self.shuffle = ds, batch_size, shuffle
        self.rng = np.random.default_rng(seed)
    def __iter__(self):
        idx = np.arange(len(self.ds))
        if self.shuffle:
            self.rng.shuffle(idx)
        for i in range(0, len(idx), self.bs):
            sel = idx[i:i+self.bs]
            yield self.ds.X[sel], self.ds.y[sel]

rng = np.random.default_rng(42)
n = 400
X0 = rng.standard_normal((n, 2)) + np.array([-2.0, 0.0])
X1 = rng.standard_normal((n, 2)) + np.array([2.0, 0.0])
X = np.vstack([X0, X1]); y = np.concatenate([np.zeros(n), np.ones(n)])
idx = rng.permutation(len(X)); tr, va = idx[:320], idx[320:]

train_ds = Dataset(X[tr], y[tr]); val_ds = Dataset(X[va], y[va])
train_dl = DataLoader(train_ds, 32, shuffle=True)
print("首个 batch 形状:", next(iter(train_dl))[0].shape, "→ (batch, 特征数)")
"""))

c.append(md("""## 2. 训练器：Trainer
"""))
c.append(code("""class Trainer:
    def __init__(self, model, opt, train_dl, val_dl=None, epochs=50):
        self.model, self.opt = model, opt
        self.train_dl, self.val_dl = train_dl, val_dl
        self.epochs = epochs
        self.history = {'train': [], 'val': [], 'val_acc': []}

    def fit(self):
        for epoch in range(self.epochs):
            self.model.train()
            ep = []
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                loss = F.binary_cross_entropy(torch.sigmoid(self.model(xb)), yb)
                loss.backward(); self.opt.step()
                ep.append(loss.item())
            self.history['train'].append(np.mean(ep))
            self.evaluate()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            out = torch.sigmoid(self.model(self.val_ds.X))
            loss = F.binary_cross_entropy(out, self.val_ds.y).item()
            acc = ((out.ravel() > 0.5) == self.val_ds.y.ravel()).float().mean().item()
        self.history['val'].append(loss); self.history['val_acc'].append(acc)

    def __getattr__(self, name):
        # 便捷访问：trainer.val_ds
        if name == 'val_ds' and self.val_dl is not None:
            return self.val_dl.ds
        raise AttributeError(name)
"""))

c.append(md("""## 3. 用同一 Trainer 训练不同模型
"""))
c.append(code("""def train_and_report(make_model, make_opt, name):
    torch.manual_seed(0)
    trainer = Trainer(make_model(), make_opt, train_dl, DataLoader(val_ds, 32, shuffle=False), epochs=80)
    trainer.fit()
    acc = trainer.history['val_acc'][-1]
    print(f"{name:<14} val 准确率 = {acc:.3f}")
    return trainer

# 同一个 Trainer，换模型/优化器只需改构造参数
t_mlp  = train_and_report(lambda: nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1)),
                          lambda p: torch.optim.Adam(p, lr=0.01), 'MLP + Adam')
t_mlp_sgd = train_and_report(lambda: nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1)),
                          lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9), 'MLP + Momentum')
t_linear = train_and_report(lambda: nn.Linear(2, 1),
                          lambda p: torch.optim.Adam(p, lr=0.01), 'Linear + Adam')

plt.figure(figsize=(8, 4.5))
for t, name in [(t_mlp, 'MLP+Adam'), (t_mlp_sgd, 'MLP+Momentum'), (t_linear, 'Linear+Adam')]:
    plt.plot(t.history['val_acc'], label=name)
plt.xlabel('epoch'); plt.ylabel('val 准确率')
plt.title('同一 Trainer、不同模型/优化器的可复用对比')
plt.legend(); plt.grid(alpha=0.3)
"""))

c.append(md("""## 4. 扩展点：换损失、加正则
"""))
c.append(code("""# 扩展 1：换损失函数（MSE 用于回归任务）
class RegTrainer(Trainer):
    def fit(self):
        for epoch in range(self.epochs):
            self.model.train()
            ep = []
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                loss = F.mse_loss(self.model(xb), yb)          # 只改这一行
                loss.backward(); self.opt.step()
                ep.append(loss.item())
            self.history['train'].append(np.mean(ep))
            self.evaluate()

# 扩展 2：L2 正则（weight_decay）
torch.manual_seed(0)
reg_model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
t_reg = Trainer(reg_model,
                torch.optim.Adam(reg_model.parameters(), lr=0.01, weight_decay=1e-3),
                train_dl, DataLoader(val_ds, 32, shuffle=False), epochs=80)
t_reg.fit()
print("加 weight_decay 的 val 准确率:", round(t_reg.history['val_acc'][-1], 3))
"""))

c.append(md("""## 5. 从数学到框架的完整链路
"""))
c.append(md("""回顾整个课程链：线性代数（矩阵如何表示变换）→ 微积分（梯度从哪来）→ 概率（损失 = 负对数似然）→ 神经网络（把这三者装进一个可训练的系统）。本课的 Trainer 就是最小化的 `PyTorch Lightning` / 自定义训练循环。

**工程化方向**（超出本课范围）：`torch.utils.data.DataLoader`（多进程加载）、`torch.compile`（加速）、分布式训练、checkpoint 与日志。
"""))

c.append(md("""## 课后练习
"""))
c.append(md("""1. **加 Early Stopping**：给 Trainer 加 `patience` 参数，val 不降则停止并恢复最优权重。
2. **加 Dropout**：构造带 Dropout 的模型用 Trainer 训练，对比泛化。
3. **回归任务**：造一个 $y = \\sin(x) + \\text{噪声}$ 的回归数据集，用 RegTrainer 训练并画拟合曲线。
4. **换优化器**：给 Trainer 加 `scheduler` 支持（每个 epoch 后 step）。
5. **思考**：Trainer 的哪些部分应该抽象、哪些应该暴露？对比你见过的训练框架设计。
"""))

write_nb("08-mini-training-framework", "mini-training-framework.ipynb", c)
write_readme("08-mini-training-framework", """# 08 · Mini Training Framework

综合：手写 Dataset/DataLoader/Trainer，同一框架训练 MLP 与线性模型、换损失函数、加正则，回顾"从数学到框架"的完整链路。

## 前置要求

- 完成本模块 01-07 全部课程

## 课程内容

| 章节 | 内容 |
|------|------|
| 1 | Dataset 与 DataLoader |
| 2 | Trainer 训练器 |
| 3 | 同一 Trainer 训练不同模型 |
| 4 | 扩展点：换损失/加正则 |
| 5 | 从数学到框架的完整链路 |
| - | 课后练习 |

## 学习建议

- 本课是 nn-core 的"毕业设计"：把前七课全部装进一个可复用框架
- 对比 `torch.utils.data` 的官方实现，理解每个抽象层解决什么问题
""")
