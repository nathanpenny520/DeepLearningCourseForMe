# 应用机器学习模块（08 Applied ML）

把前七个模块的算法能力，落成"拿真实数据训练一个能用的模型"的完整工程链路。本模块聚焦**工程层**：数据获取与清洗、基线模型、训练工程、超参调优、评估与错误分析、部署、监控，以及一个端到端 Kaggle 风格实战。

> 本模块不需要 GPU，也不需要 PyTorch——全部用 numpy / pandas / scikit-learn 完成，重点是把工程细节讲透。算法原理若需回顾，回看前序模块。

## 课程目录

| # | 课程 | 文件夹 | 状态 |
|---|------|--------|------|
| 01 | 数据工程基础（真实数据获取、清洗、划分） | [01-data-engineering](./01-data-engineering) | ✅ |
| 02 | 基线与表格模型（线性 / 树模型对比，何时上 DL） | [02-baselines-tabular](./02-baselines-tabular) | ✅ |
| 03 | 训练工程（seed / checkpoint / 早停 / LR 调度 / 梯度裁剪） | [03-training-engineering](./03-training-engineering) | ✅ |
| 04 | 超参调优与实验管理（搜索、交叉验证、学习曲线、实验记录） | [04-hyperparameter-tuning](./04-hyperparameter-tuning) | ✅ |
| 05 | 评估与错误分析（指标、校准、错例分析闭环） | [05-evaluation-error-analysis](./05-evaluation-error-analysis) | ✅ |
| 06 | 部署与服务化（导出、预测接口、FastAPI、版本管理） | [06-deployment-serving](./06-deployment-serving) | ✅ |
| 07 | 监控与数据漂移（PSI / KS、三路监控、重训触发） | [07-monitoring-drift](./07-monitoring-drift) | ✅ |
| 08 | 端到端实战（Kaggle 工作流：EDA→特征→基线→调参→提交） | [08-end-to-end-project](./08-end-to-end-project) | ✅ |

## 环境配置

本模块所有课程共用一套依赖与虚拟环境，完整配置步骤见 [`guide.md`](./guide.md)，注册内核名为 **Python (applied-ml)**。

## 设计原则

- **真实数据优先**：01/02/08 使用公开 Titanic 数据集（GitHub 镜像，无需 Kaggle 账号）；课后练习给出 Kaggle 原竞赛链接与达标分数。
- **工程细节可复现**：每个环节都有可运行的微型实现（断点续训、PSI 检测、提交文件生成等）。
- **先基线后深度**：02 课专门讲"表格问题先跑树模型基线，再决定要不要上深度模型"的决策框架。