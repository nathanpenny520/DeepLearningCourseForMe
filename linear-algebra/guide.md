## 环境配置

> 推荐 Python 3.12（不建议使用最新版如 3.14，深度学习库可能尚未兼容）。
> 工作目录：本模块根目录 `linear-algebra/`（即本文件所在目录）

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

### 在 VS Code 中使用 Notebook（无需注册内核）

VS Code 的 Jupyter 扩展可以**直接使用虚拟环境中的 Python 解释器**，不需要执行 `ipykernel install` 注册内核。

打开 `.ipynb` 文件后：

1. 点击右上角的内核选择器（或 `Cmd+Shift+P` → 输入 `Notebook: Select Notebook Kernel`）
2. 选择 **Select Another Kernel...** → **Python Environments...**
3. 选择工作目录下的 `venv`（通常显示为 `venv/bin/python`）

选择后内核选择器会显示类似 `venv (3.12.x) venv/bin/python`，表示 VS Code 已直连虚拟环境运行 notebook。

> 原理：VS Code 直接调用 `venv/bin/python -m ipykernel_launcher` 来启动内核，绕过了 Jupyter 的 kernelspec 注册表。只要虚拟环境里装了 `ipykernel`（`requirements.txt` 已包含），就能直接用。

---

### 方案二：传统 venv + pip（备选）

如果不使用 uv，可以用标准库 venv：

```bash
# 1. 安装 Python 3.12（如未安装）
brew install python@3.12

# 2. 进入项目目录
cd linear-algebra

# 3. 创建虚拟环境
python3.12 -m venv venv

# 4. 激活
source venv/bin/activate

# 5. 安装依赖
pip install -r requirements.txt
```

---

### 如果你使用 JupyterLab / Jupyter Notebook（网页版）

网页版 Jupyter 依赖 kernelspec 注册表来发现内核，因此需要额外注册一步：

```bash
source venv/bin/activate
python -m ipykernel install --user --name=dl_venv --display-name="Python (dl-demo-torch)"
```

注册成功输出示例：

```
Installed kernelspec dl_venv in /Users/yourname/Library/Jupyter/kernels/dl_venv
```

启动：

```bash
jupyter notebook
```

在网页界面中通过 **Kernel → Change Kernel** 选择 `Python (dl-demo-torch)`。

#### 内核维护命令

```bash
# 查看已注册的 Jupyter 内核列表
jupyter kernelspec list

# 删除注册的内核（仅移除 Jupyter 内核记录，不会删除本地 venv 文件夹）
jupyter kernelspec remove dl_venv
```

> 提示：注册内核只需执行一次；VS Code 用户可跳过此步骤。
