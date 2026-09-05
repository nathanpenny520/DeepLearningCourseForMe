## 环境配置

> 推荐 Python 3.12（不建议使用最新版如 3.14，深度学习库可能尚未兼容）。
> 以下命令均在**仓库根目录**下开始执行，先进入模块目录 `cd probability`。
> 课程代码本身跨平台，仅安装和激活命令因系统而异。

**首选方案：uv**（极速 Python 包管理器，自动管理 Python 版本）。
传统 venv + pip 方案见文末备选。

---

### 方案一：uv（推荐）

#### 1. 安装 uv

| 系统 | 命令 |
|------|------|
| Mac | `brew install uv` |
| Windows | `winget install astral-sh.uv`（或 `pip install uv`） |

安装后验证：`uv --version`

#### 2. 创建虚拟环境（跨平台相同）

uv 会自动下载 Python 3.12，无需提前安装。

```bash
cd probability
uv venv venv --python 3.12
```

#### 3. 激活虚拟环境

| 系统 | 命令 |
|------|------|
| Mac | `source venv/bin/activate` |
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |

> Windows 若提示执行策略禁止，先运行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

#### 4. 安装依赖（跨平台相同）

```bash
uv pip install -r requirements.txt
```

---

### 注册 Jupyter 内核（推荐执行一次，跨平台相同）

由于虚拟环境位于模块子目录，VS Code 自动搜索可能发现不了。注册后 VS Code 和网页版 Jupyter 都能直接选择。

```bash
# 确保已激活 venv
python -m ipykernel install --user --name=probability --display-name="Python (probability)"
```

> 每个模块只需注册一次。新增其他模块时用不同的 `--name` 重复即可。

---

### 在 VS Code 中使用 Notebook

打开 `01-random-variables-distributions/01-random-variables-distributions.ipynb`：

1. 点击右上角内核选择器（Mac: `Cmd+Shift+P`，Windows: `Ctrl+Shift+P` → `Notebook: Select Notebook Kernel`）
2. 选 **Select Another Kernel...** → **Jupyter Kernel...**（不是 Python Environments...）
3. 选择 **Python (probability)**

> 关键区别：`Python Environments...` 是 VS Code 自动扫描的解释器，发现不了子目录的 venv；`Jupyter Kernel...` 是从注册表读取的，注册的内核在这里。

#### 如果列表里没有

1. 重载窗口：`Cmd+Shift+P` / `Ctrl+Shift+P` → `Developer: Reload Window`
2. 仍没有则手动指定：`Python Environments...` → `Enter interpreter path...` → 浏览到 `probability/venv/bin/python`（Mac）或 `probability\venv\Scripts\python.exe`（Windows）

---

### 方案二：传统 venv + pip（备选）

#### 1. 安装 Python 3.12

| 系统 | 方式 |
|------|------|
| Mac | `brew install python@3.12` |
| Windows | 从 [python.org](https://www.python.org/downloads/) 下载安装，勾选 "Add Python to PATH" |

#### 2. 创建并激活虚拟环境

```bash
cd probability
```

| 系统 | 创建 | 激活 |
|------|------|------|
| Mac | `python3.12 -m venv venv` | `source venv/bin/activate` |
| Windows | `python -m venv venv` | `venv\Scripts\Activate.ps1` |

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

安装完成后同样执行上面的「注册 Jupyter 内核」步骤。

---

### 网页版 Jupyter（可选）

注册内核后启动：

```bash
jupyter notebook
```

在网页中 **Kernel → Change Kernel** 选择 `Python (probability)`。

---

### 内核维护命令（跨平台相同）

```bash
# 查看已注册内核
jupyter kernelspec list

# 删除内核（仅移除注册记录，不删除 venv）
jupyter kernelspec remove probability
```
