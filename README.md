# Deep Learning Course For Me

个人深度学习系列课程笔记与实战代码。每堂课一个独立文件夹，自带环境与说明，可单独运行。

## 课程目录

| # | 课程 | 文件夹 | 状态 |
|---|------|--------|------|
| 1 | 深度学习中的线性代数：PyTorch 实战 | [linear-algebra](./linear-algebra) | ✅ |

## 每堂课的通用结构

```
course-name/
├── README.md          # 本课说明与内容大纲
├── guide.md           # 环境配置指南（可选）
├── requirements.txt   # Python 依赖（已锁定版本）
└── *.ipynb            # 教学代码（Jupyter Notebook）
```

每堂课使用独立虚拟环境，进入对应文件夹后按其 `README.md` / `guide.md` 配置即可。

## 通用环境要求

- Python 3.12
- 推荐包管理器：[uv](https://docs.astral.sh/uv/)（也支持传统 venv + pip）
- 编辑器：VS Code + Jupyter 扩展（或网页版 Jupyter Notebook）
