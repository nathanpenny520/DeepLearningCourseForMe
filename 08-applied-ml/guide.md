## 环境配置

> 本模块只需 CPU 即可运行（numpy / pandas / scikit-learn），不需要 PyTorch 与 GPU，推荐 Python 3.12+。
> 以下命令均在**仓库根目录**下开始执行，先进入模块目录 `cd 08-applied-ml`。

**首选方案：venv + pip**；若已安装 [uv](https://docs.astral.sh/uv/) 也可用文末的 uv 方案。

---

### 方案一：venv + pip（推荐）

#### 1. 创建并激活虚拟环境

| 系统 | 创建 | 激活 |
|------|------|------|
| Windows (PowerShell) | `python -m venv venv` | `venv\Scripts\Activate.ps1` |
| macOS / Linux | `python3 -m venv venv` | `source venv/bin/activate` |

> Windows 若提示执行策略禁止，先运行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

---

### 方案二：uv（若已安装 uv）

```bash
cd 08-applied-ml
uv venv venv --python 3.12
uv pip install -r requirements.txt
```

> uv 方案同样需要激活虚拟环境，激活命令见方案一。

---

### 注册 Jupyter 内核（推荐执行一次）

由于虚拟环境位于模块子目录，VS Code 自动搜索可能发现不了。注册后 VS Code 和网页版 Jupyter 都能直接选择。

```bash
# 确保已激活 venv
python -m ipykernel install --user --name=applied-ml --display-name="Python (applied-ml)"
```

> 每个模块只需注册一次。

---

### 在 VS Code 中使用 Notebook

打开任意课程的 `.ipynb`：

1. 点击右上角内核选择器（Windows: `Ctrl+Shift+P` → `Notebook: Select Notebook Kernel`）
2. 选 **Select Another Kernel...** → **Jupyter Kernel...**（不是 Python Environments...）
3. 选择 **Python (applied-ml)**

> 关键区别：`Python Environments...` 是 VS Code 自动扫描的解释器，发现不了子目录的 venv；`Jupyter Kernel...` 是从注册表读取的，注册的内核在这里。

#### 如果列表里没有

1. 重载窗口：`Ctrl+Shift+P` → `Developer: Reload Window`
2. 仍没有则手动指定：`Python Environments...` → `Enter interpreter path...` → 浏览到 `08-applied-ml\venv\Scripts\python.exe`（Windows）或 `08-applied-ml/venv/bin/python`（macOS / Linux）

---

### 网页版 Jupyter（可选）

注册内核后启动：

```bash
jupyter notebook
```

在网页中 **Kernel → Change Kernel** 选择 `Python (applied-ml)`。

---

### 内核维护命令

```bash
# 查看已注册内核
jupyter kernelspec list

# 删除内核（仅移除注册记录，不删除 venv）
jupyter kernelspec remove applied-ml
```

---

### 数据说明

- 01/02/08 课会在首次运行时自动下载公开的 Titanic 数据集（GitHub 镜像：`raw.githubusercontent.com/datasciedojo/datasets/master/titanic.csv`）。
- 若网络不可用，课程代码会自动回退到 scikit-learn 内置数据集（如 breast_cancer），保证所有 notebook 在任何环境都能跑通。
- 课后练习涉及的 Kaggle 竞赛（Titanic / House Prices / Digit Recognizer / Porto Seguro）需要 Kaggle 账号在线完成，见各课 README 中的链接。