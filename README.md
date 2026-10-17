# RL-TDAGCN: 拓扑感知动态自适应图卷积网络

> **R**einforcement **L**earning-enhanced **T**opology-aware **D**ynamic **A**daptive **G**raph **C**onvolutional **N**etwork for Traffic Prediction

---

## 📌 项目简介

RL-TDAGCN 是一种融合**强化学习图控制**与**拓扑约束**的动态自适应图卷积网络，用于交通拥堵预测。该方法针对现有动态图学习中存在的**结构漂移**和**更新策略缺乏优化**两大问题，提出了系统性的解决方案。

### 核心创新点

1. **双重拓扑约束机制** — 加权MSE拓扑损失 + 可学习β融合，确保动态图不偏离路网物理结构
2. **PPO强化学习图控制器** — 混合动作空间（离散更新频率 + 连续融合比例），自适应决策图更新策略
3. **两阶段训练策略** — 主干预训练 + RL交替优化，确保训练稳定性

### 实验结果（PEMS04 数据集）

| 变体 | MAE ↓ | RMSE ↓ | SMAPE ↓ | 相对改善 |
|------|-------|--------|---------|---------|
| DAGCN（静态图基线） | 2.0102 | 4.3286 | 3.89% | --- |
| Base（纯动态图） | 2.0032 | 4.3324 | 3.87% | +0.3% |
| + Topo（拓扑约束） | **1.8945** | **4.0884** | **3.67%** | **+5.8%** |
| + RL + Topo（完整模型） | 1.9301 | 4.1929 | 3.79% | +4.0% |

> **关键发现**：拓扑约束是性能提升的主要来源（-5.4% MAE）。RL 在 PEMS04 小规模数据集上效果有限，推测其优势在大规模、复杂模式的数据集上更为显著。

---

## 📁 项目结构

```
RL-TDAGCN/
├── models/                    # 核心模型代码
│   ├── rlt_dagcn.py          # RL-TDAGCN 主模型
│   ├── base_encoder.py       # 基础编码器 (GRU + 扩散卷积)
│   ├── dynamic_graph.py      # 动态图生成器 (多头注意力)
│   ├── rl_graph_controller.py # PPO 强化学习图控制器
│   ├── topology_constraint.py # 拓扑约束模块
│   ├── multimodal_fusion.py  # 多模态天气融合
│   └── baselines.py          # 基线模型 (LSTM/STGCN/TGCN/DCRNN/GWN)
├── configs/
│   └── default.py            # 默认超参数配置
├── data/
│   └── PEMS04/               # PEMS04 数据集 (需自行下载)
│       ├── PEMS04.csv
│       └── PEMS04_inner.npz
├── experiments/
│   ├── run_experiments.py    # 消融实验运行脚本
│   └── results_v5/           # v5 版本实验结果 (JSON)
├── utils/
│   ├── dataloader.py         # 数据加载器
│   ├── metrics.py            # 评价指标 (MAE/RMSE/SMAPE)
│   ├── early_stopping.py     # 早停机制
│   └── visualization.py      # 可视化工具
├── train.py                  # 主训练入口
├── requirements.txt          # Python 依赖
└── README.md
```

---

## 🚀 快速开始

### 环境要求

- Python >= 3.9
- PyTorch >= 1.12
- CUDA >= 11.3 (推荐 GPU 训练)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

1. 下载 PEMS04 数据集
2. 放置到 `data/PEMS04/` 目录下，包含：
   - `PEMS04.csv` — 原始交通数据
   - `PEMS04_inner.npz` — 预处理后的数据

### 训练模型

```bash
# 训练完整模型 (RL + Topo + 动态图)
python train.py --model full --dataset PEMS04

# 训练基线模型
python train.py --model dagcn --dataset PEMS04     # 静态图 DAGCN
python train.py --model base --dataset PEMS04      # 纯动态图
python train.py --model topo --dataset PEMS04      # 动态图 + 拓扑约束
python train.py --model rl --dataset PEMS04        # 动态图 + RL
```

### 运行消融实验

```bash
# 运行所有 5 组消融实验
python experiments/run_experiments.py

# 查看结果
# 结果保存在 experiments/results_v5/exp{1-5}_results.json
```

---

## 🧩 模型架构

```
输入 [T, N, F]
    │
    ├─→ 第一阶段编码 (扩散卷积 + GRU)
    │       │
    │       ├─→ 动态图生成器 (多头注意力) → A_candidate
    │       │
    │       ├─→ PPO RL 图控制器 ────────────→ 状态/动作/奖励
    │       │                                    │
    │       │                                    ▼
    │       └─→ 拓扑约束模块 (GCN + β融合) → A_final
    │
    ├─→ 多模态融合 (可选：天气特征)
    │
    ├─→ 第二阶段空间精炼 (A_final 上的扩散卷积)
    │
    └─→ 预测输出 [P, N]
```

### 关键模块说明

| 模块 | 文件 | 说明 |
|------|------|------|
| 动态图生成器 | `dynamic_graph.py` | 4头注意力，温度缩放softmax，对称化 |
| RL控制器 | `rl_graph_controller.py` | PPO算法，6维状态，混合动作空间 |
| 拓扑约束 | `topology_constraint.py` | 3层GCN拓扑编码器，加权MSE损失，可学习β |
| 扩散卷积 | `base_encoder.py` | K=2阶，前向+后向双向扩散 |
| 多模态融合 | `multimodal_fusion.py` | 门控自适应融合 |

---

## 📊 消融实验设计

| 变体 | 动态图 | RL控制 | 拓扑约束 | 多模态 |
|------|:------:|:------:|:--------:|:------:|
| DAGCN | ✗ | ✗ | ✗ | ✗ |
| Base | ✓ | ✗ | ✗ | ✗ |
| + RL | ✓ | ✓ | ✗ | ✗ |
| + Topo | ✓ | ✗ | ✓ | ✗ |
| + RL + Topo (Full) | ✓ | ✓ | ✓ | ✗ |
| + Weather | ✓ | ✓ | ✓ | ✓ |

---

## 📝 主要超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_dim | 64 | 隐藏层维度 |
| diffusion_steps | 2 | 扩散阶数 |
| num_heads | 4 | 注意力头数 |
| embed_dim | 10 | 节点嵌入维度 |
| batch_size | 64 | 批大小 |
| learning_rate | 1e-3 | 学习率 |
| dropout | 0.3 | Dropout率 |
| epochs | 150 | 最大训练轮数 |
| rl_warmup_epochs | 30 | RL开始更新的预热轮数 |
| ppo_clip_eps | 0.1 | PPO裁剪范围 |
| topo_loss_weight | 0.05 | 拓扑损失权重 |
| beta_init | 0.6 | 动态图/拓扑图融合初始比例 |

---

## 🔬 机理分析（待补充）

计划中的机理分析包括四个维度：

1. **图结构差异分析** — 度分布、权重分布、与静态图偏离度
2. **图谱频域分析** — 拉普拉斯特征值分布、低/中/高频能量占比
3. **相变行为分析** — β参数扫描、U型相变曲线
4. **RL策略可解释性** — 状态重要性排名、动作分布分析

---

## 📄 论文

相关论文正在撰写中，详细内容请参考论文文档。

---

## 📜 许可证

本项目仅供学术研究使用。

---

## 📮 联系方式

如有问题或建议，欢迎提交 Issue 或联系作者。
