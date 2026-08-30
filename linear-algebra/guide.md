## 环境配置

> 推荐 Python 3.12（不建议使用最新版如 3.14，深度学习库可能尚未兼容）。
> 以下所有命令均在**仓库根目录**下开始执行，先进入模块目录 `cd linear-algebra`。

**首选方案：uv**（Rust 编写的极速 Python 包管理器，替代 venv + pip）。
传统 venv + pip 方案见文末备选。

---

### 方案一：uv（推荐）

#### 1. 安装 uv

```bash
brew install uv
```

或官方脚本：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后验证：

```bash
uv --version
```

#### 2. 创建虚拟环境

```bash
# 在仓库根目录下执行
cd linear-algebra

# 创建 venv，指定 Python 3.12
# uv 会自动下载并管理 Python 版本，无需提前安装 Python 3.12
uv venv venv --python 3.12
```

#### 3. 激活虚拟环境

```bash
source venv/bin/activate
```

#### 4. 安装依赖

```bash
uv pip install -r requirements.txt
```

> uv 的 `pip` 子命令比传统 pip 快 10-100 倍，且解析依赖更可靠。

---

### 注册 Jupyter 内核（推荐执行一次）

由于虚拟环境位于模块子目录（`linear-algebra/venv/`），VS Code 的自动搜索可能发现不了它。**注册内核后，VS Code 和网页版 Jupyter 都能在内核列表中直接选择，一劳永逸。**

```bash
# 确保已激活 venv
source venv/bin/activate
python -m ipykernel install --user --name=linear-algebra --display-name="Python (linear-algebra)"
```

注册成功输出示例：

```
Installed kernelspec linear-algebra in /Users/yourname/Library/Jupyter/kernels/linear-algebra
```

> 每个模块只需注册一次。如果之后新增了其他模块（如 calculus），在对应模块的 venv 中用不同的 `--name` 重复此步骤即可。

---

### 在 VS Code 中使用 Notebook

打开 `01-tensor-basics/tensor-basics.ipynb` 后：

1. 点击右上角的内核选择器（或 `Cmd+Shift+P` → 输入 `Notebook: Select Notebook Kernel`）
2. 选择 **Select Another Kernel...**
3. **选择 Jupyter Kernel...**（不是 Python Environments...）
4. 在列表中选择 **Python (linear-algebra)**

选择后内核选择器会显示 `Python (linear-algebra)`，表示已连接到虚拟环境。

> **关键区别**：`Python Environments...` 是 VS Code 自动扫描的解释器，发现不了子目录里的 venv；`Jupyter Kernel...` 是从 Jupyter 注册表读取的，我们刚才注册的内核就在这里。

#### 如果列表里没有 Python (linear-algebra)

1. 先重载 VS Code 窗口：`Cmd+Shift+P` → `Developer: Reload Window`
2. 重载后重复上面的步骤，注册的内核应该就出现了
3. 如果还是没有，用手动方式：第 3 步改选 **Python Environments...** → **Enter interpreter path...** → **Browse...**，导航到 `linear-algebra/venv/bin/python`

#### 内核与虚拟环境的关系

- **虚拟环境（venv）**：实际运行代码的 Python 环境，包含 torch、numpy 等所有安装的包
- **Jupyter 内核**：只是一个注册记录（`kernel.json`），告诉 Jupyter/VS Code "去调用这个 venv 里的 Python"
- **注册内核不会安装任何包**，它只是创建一个指向 venv 的快捷方式
- 删除内核（`jupyter kernelspec remove`）只会删除注册记录，不会删除 venv 里的任何东西

---

### 方案二：传统 venv + pip（备选）

如果不使用 uv，可以用标准库 venv：

```bash
# 1. 安装 Python 3.12（如未安装）
brew install python@3.12

# 2. 进入模块目录（在仓库根目录下执行）
cd linear-algebra

# 3. 创建虚拟环境
python3.12 -m venv venv

# 4. 激活
source venv/bin/activate

# 5. 安装依赖
pip install -r requirements.txt
```

安装完成后，同样需要执行上面的「注册 Jupyter 内核」步骤。

---

### 如果你使用 JupyterLab / Jupyter Notebook（网页版）

注册内核后，启动网页版：

```bash
source venv/bin/activate
jupyter notebook
```

在网页界面中通过 **Kernel → Change Kernel** 选择 `Python (linear-algebra)`。

---

### 内核维护命令

```bash
# 查看已注册的 Jupyter 内核列表
jupyter kernelspec list

# 删除注册的内核（仅移除 Jupyter 内核记录，不会删除本地 venv 文件夹）
jupyter kernelspec remove linear-algebra
```
