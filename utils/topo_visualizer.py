"""
拓扑可视化工具 (Topology Visualizer)
用于诊断拓扑约束模块的效果，生成邻接矩阵热力图对比与beta变化曲线。

包含:
  1. plot_adjacency_heatmap  - 单张邻接矩阵热力图
  2. plot_beta_trajectory    - beta值在训练过程中的变化曲线
  3. plot_topo_comparison    - 动态图/拓扑图/融合图 三图并排对比

字体设置: 优先使用 SimHei (Windows中文)，依次回退 Microsoft YaHei /
Noto Sans CJK SC / DejaVu Sans，确保中文标题正常显示。
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，适合服务器/脚本环境
import matplotlib.pyplot as plt

# 中文字体设置 (SimHei 为 Windows 常用中文字体，DejaVu Sans 为兜底)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def _to_numpy(x):
    """
    将输入统一转为 numpy.ndarray

    支持 torch.Tensor / numpy.ndarray / list，
    便于可视化函数统一处理多种输入类型。

    Args:
        x: torch.Tensor / numpy.ndarray / list

    Returns:
        np.ndarray
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def plot_adjacency_heatmap(adj_matrix, title, save_path, cmap='YlOrRd'):
    """
    绘制单张邻接矩阵热力图

    Args:
        adj_matrix: [N, N] 邻接矩阵 (torch.Tensor / numpy.ndarray 均可)
        title:      图标题
        save_path:  保存路径 (含文件名)
        cmap:       颜色映射，默认 'YlOrRd' (黄-橙-红)
    """
    adj = _to_numpy(adj_matrix)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(adj, cmap=cmap, aspect='equal')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('节点索引')
    ax.set_ylabel('节点索引')
    plt.colorbar(im, ax=ax, label='边权重')
    plt.tight_layout()

    # 确保目录存在 (save_path 可能不含目录分隔符，取绝对路径的 dirname)
    dir_path = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[TopoVis] 保存邻接矩阵热力图: {save_path}')


def plot_beta_trajectory(beta_history, save_path):
    """
    绘制beta值在训练过程中的变化曲线

    用于监控可学习融合参数 beta 的演化趋势:
      - beta 接近1: 模型更依赖动态图 (数据驱动)
      - beta 接近0: 模型更依赖拓扑结构图 (道路先验)

    Args:
        beta_history: list of float, 每个epoch/sample记录的beta值
        save_path:    保存路径
    """
    beta_history = list(beta_history)
    if len(beta_history) == 0:
        print('[TopoVis] beta_history 为空，跳过绘制')
        return

    steps = np.arange(1, len(beta_history) + 1)
    betas = np.array(beta_history, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, betas, color='steelblue', linewidth=1.5, marker='o', markersize=3)
    # 参考线: beta=0.5 表示动态图与拓扑图各占一半
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='beta=0.5 (均衡)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('记录步数')
    ax.set_ylabel('beta 值')
    ax.set_title('拓扑融合参数 beta 训练曲线', fontsize=13)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    dir_path = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[TopoVis] 保存beta变化曲线: {save_path}')


def plot_topo_comparison(dynamic_adj, topo_adj, final_adj, epoch, save_dir):
    """
    并排对比三张邻接矩阵: 动态图、拓扑图、最终融合图

    用于诊断拓扑约束是否有效融合了道路结构先验:
      - 若动态图与拓扑图差异很大，但融合图接近动态图，说明 beta 偏大、约束弱
      - 若融合图明显向拓扑图靠拢，说明拓扑约束正在发挥作用

    Args:
        dynamic_adj: [N, N] 动态图 (拓扑约束前)
        topo_adj:    [N, N] 拓扑引导图
        final_adj:   [N, N] 拓扑融合后的最终图
        epoch:       当前训练轮次 (用于标题与文件名)
        save_dir:    保存目录 (文件名为 topo_epoch_{epoch}.png)
    """
    # 统一转 numpy
    dyn = _to_numpy(dynamic_adj)
    topo = _to_numpy(topo_adj)
    final = _to_numpy(final_adj)

    # 三图并排
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    titles = ['动态图 (Dynamic)', '拓扑引导图 (Topo)', '最终融合图 (Final)']
    matrices = [dyn, topo, final]

    # 统一颜色范围，便于横向比较三张图的数值差异
    vmax = max(m.max() for m in matrices)
    vmin = min(m.min() for m in matrices)

    im = None
    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(mat, cmap='YlOrRd', aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('节点索引')
        ax.set_ylabel('节点索引')

    # 共享一个 colorbar (基于最后绘制的 im，因 vmin/vmax 已统一)
    fig.colorbar(im, ax=axes, shrink=0.8, label='边权重')
    fig.suptitle(f'拓扑约束诊断 - Epoch {epoch}', fontsize=14, y=1.02)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'topo_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[TopoVis] 保存拓扑对比图: {save_path}')
