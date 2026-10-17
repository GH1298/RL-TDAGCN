"""
可视化工具
用于生成论文所需的图表:
  1. 动态图演化热力图 - 展示不同时段动态图结构变化
  2. RL 动作曲线 - 展示RL控制器在训练过程中的决策
  3. 训练损失曲线 - 展示模型收敛过程
  4. 消融实验对比柱状图 - 展示各消融变体的性能
  5. 分步预测性能曲线 - 展示不同预测步长的指标变化

使用 matplotlib，字体设置为 Noto Sans CJK SC（中文字体）。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_graph_heatmap(adj_matrix, save_path, title='Dynamic Graph',
                       node_indices=None, figsize=(8, 6)):
    """
    动态图热力图

    Args:
        adj_matrix: (N, N) 邻接矩阵
        save_path: 保存路径
        title: 标题
        node_indices: 显示的节点子集索引（大规模图时截取子图）
        figsize: 图像大小
    """
    if node_indices is not None:
        adj = adj_matrix[np.ix_(node_indices, node_indices)]
    else:
        adj = adj_matrix

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(adj, cmap='YlOrRd', aspect='equal')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Node Index')
    plt.colorbar(im, ax=ax, label='Edge Weight')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] 保存图热力图: {save_path}")


def plot_graph_evolution(adj_list, save_path, titles=None,
                         node_indices=None, n_cols=4, figsize_per=3):
    """
    动态图演化热力图（多时刻对比）

    Args:
        adj_list: list of (N, N) 邻接矩阵
        save_path: 保存路径
        titles: 每个子图标题
        node_indices: 节点子集
        n_cols: 列数
        figsize_per: 每个子图的大小
    """
    n = len(adj_list)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    if node_indices is not None:
        adjs = [a[np.ix_(node_indices, node_indices)] for a in adj_list]
    else:
        adjs = adj_list

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per * n_cols, figsize_per * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    vmax = max(a.max() for a in adjs)
    vmin = min(a.min() for a in adjs)

    for idx, (adj, ax) in enumerate(zip(adjs, axes.flat)):
        if idx >= n:
            ax.set_visible(False)
            continue
        im = ax.imshow(adj, cmap='YlOrRd', aspect='equal', vmin=vmin, vmax=vmax)
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)
        else:
            ax.set_title(f't={idx}', fontsize=10)
        ax.tick_params(labelsize=6)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Edge Weight')
    fig.suptitle('Dynamic Graph Evolution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] 保存图演化热力图: {save_path}")


def plot_rl_actions(rl_log, save_path):
    """
    RL 动作曲线

    Args:
        rl_log: list of dict, 每个dict包含:
            - 'step': 步数
            - 'freq': 离散动作 (0/1/2)
            - 'alpha': 连续动作 (0~1)
            - 'reward': 奖励
        save_path: 保存路径
    """
    steps = [x['step'] for x in rl_log]
    freqs = [x['freq'] for x in rl_log]
    alphas = [x['alpha'] for x in rl_log]
    rewards = [x['reward'] for x in rl_log]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 更新频率
    axes[0].scatter(steps, freqs, c='steelblue', s=10, alpha=0.6)
    axes[0].set_ylabel('Update Freq')
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(['No Update', 'Normal', 'Fast'])
    axes[0].set_title('RL Action: Update Frequency')

    # 更新比例
    axes[1].plot(steps, alphas, c='darkorange', alpha=0.7, linewidth=0.8)
    axes[1].set_ylabel('Alpha (update ratio)')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title('RL Action: Update Ratio')

    # 奖励
    axes[2].plot(steps, rewards, c='green', alpha=0.7, linewidth=0.8)
    axes[2].set_ylabel('Reward')
    axes[2].set_xlabel('Step')
    axes[2].set_title('RL Reward')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] 保存RL动作曲线: {save_path}")


def plot_training_curves(train_losses, val_maes, save_path):
    """
    训练损失和验证MAE曲线

    Args:
        train_losses: list of float
        val_maes: list of float
        save_path: 保存路径
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(train_losses) + 1)
    color1 = 'steelblue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color1)
    ax1.plot(epochs, train_losses, color=color1, linewidth=1.5, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'darkorange'
    ax2.set_ylabel('Val MAE', color=color2)
    ax2.plot(epochs, val_maes, color=color2, linewidth=1.5, linestyle='--', label='Val MAE')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.title('Training Progress')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] 保存训练曲线: {save_path}")


def plot_ablation_comparison(results_dict, save_path, metric='MAE'):
    """
    消融实验对比柱状图

    Args:
        results_dict: dict, {model_name: {metric: value, ...}}
        save_path: 保存路径
        metric: 对比的指标名
    """
    names = list(results_dict.keys())
    values = [results_dict[n].get(metric, 0) for n in names]

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor='gray', linewidth=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel(metric)
    ax.set_title(f'Ablation Study: {metric}', fontsize=14)

    # 柱上标注数值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] 保存消融对比图: {save_path}")


def plot_comparison_table(results_dict, save_path):
    """
    生成对比结果表格图片（适合论文使用）

    Args:
        results_dict: dict, {model_name: {MAE: x, RMSE: y, SMAPE: z}}
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, max(3, len(results_dict) * 0.5)))
    ax.axis('off')

    # 表头
    headers = ['Model', 'MAE', 'RMSE', 'SMAPE(%)']
    table_data = [headers]

    for name, metrics in results_dict.items():
        row = [
            name,
            f"{metrics['MAE']:.4f}",
            f"{metrics['RMSE']:.4f}",
            f"{metrics['SMAPE'] * 100:.2f}",
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # 加粗表头
    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.title('Experiment Results', fontsize=14, pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Vis] 保存对比表格: {save_path}")