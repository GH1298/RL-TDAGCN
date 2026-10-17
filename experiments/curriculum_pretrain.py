"""
课程学习预训练模块 (Curriculum Pretraining)

在正式训练前，先在小规模子集(少量节点 + 少量时间步)上预训练RL控制器，
让 PPO 在小图上快速学习有效的图更新策略，再将策略迁移到大规模图上微调。

设计原理:
  - 根因分析: 全量数据(307节点)上PPO更新次数不足(100 epochs仅~18次)，
    RL控制器未充分学习即开始影响主模型，导致负优化(MAE升3%)。
  - 解决方案: 课程学习——先在50节点小图上预训练20轮(PPO更新次数充足)，
    让RL控制器学到基本的图更新策略，再迁移到大图上微调。
  - 可行性: RL控制器(PPOGraphController)是节点数无关的MLP
    (state_dim=6, hidden_dim=64)，其权重可直接从小图迁移到大图，无需维度对齐。

核心流程:
  1. 从全量训练数据中提取子集 (前sub_nodes个节点, 前sub_timesteps个时间步)
  2. 构建子图邻接矩阵 adj_static[:sub_nodes, :sub_nodes]
  3. 创建临时的小规模RLTDAGCN模型，在子集上交替训练(主模型 + RL控制器)
  4. 将临时模型的RL控制器权重复制回原始全规模模型
     (主模型参数不迁移，保持随机初始化，仅RL控制器受益于预训练)

用法:
  from experiments.curriculum_pretrain import pretrain_rl_controller
  model = pretrain_rl_controller(model, train_loader, adj_static, config, device)
"""
import os
import sys
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset

# 将项目根目录加入路径，确保能导入 models 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rlt_dagcn import RLTDAGCN


def pretrain_rl_controller(model, full_traffic_data, adj_static, config, device):
    """
    在小规模子集上预训练RL控制器

    策略:
      1. 从全量数据中提取子集 (前sub_nodes个节点, 前sub_timesteps个时间步)
      2. 构建子图的邻接矩阵
      3. 创建临时的小规模RLTDAGCN模型，在子集上交替训练
         (主模型提供预测环境，RL控制器通过PPO学习图更新策略)
      4. 将临时模型的RL控制器权重复制回原始模型
         (RL控制器是节点数无关的MLP，权重可直接迁移)

    注意:
      - 预训练阶段主模型参数也会在临时模型中训练(为RL提供有意义的预测环境)，
        但只有RL控制器权重会被迁移回原始模型，主模型保持原始初始化。
      - 奖励函数与 run_experiments.py 中的改进版保持一致
        (支持relative/absolute模式 + 图多样性奖励 + 震荡惩罚)。

    Args:
        model:            原始RLTDAGCN模型 (全规模, 如307节点)
        full_traffic_data: 训练数据 DataLoader (包含 traffic 和 target)
        adj_static:       [N, N] 静态邻接矩阵 (torch.Tensor, 已在device上)
        config:           配置对象 (configs/default.py 中的 Config 实例)
        device:           训练设备 (torch.device)

    Returns:
        model: 预训练后的原始模型 (RL控制器权重已更新，主模型不变)
    """
    # ================================================================
    # 1. 读取课程学习参数（使用 getattr 保证向后兼容）
    # ================================================================
    sub_nodes = getattr(config, 'rl_curriculum_sub_nodes', 50)          # 子集节点数
    sub_timesteps = getattr(config, 'rl_curriculum_sub_timesteps', 1000)  # 子集时间步数
    pretrain_epochs = getattr(config, 'rl_curriculum_pretrain_epochs', 20)  # 预训练轮数

    # 安全检查: 子集节点数不能超过原始节点数
    full_nodes = adj_static.size(0)
    if sub_nodes > full_nodes:
        print(f"  [课程学习] 警告: 子集节点数({sub_nodes}) > 全量({full_nodes}), "
              f"自动调整为 {full_nodes}")
        sub_nodes = full_nodes

    print(f"  [课程学习] 子集规模: {sub_nodes}节点 x {sub_timesteps}时间步, "
          f"预训练 {pretrain_epochs} 轮")

    # ================================================================
    # 2. 从训练数据中提取子集
    # ================================================================
    # 优先直接访问底层数据集（保持时间顺序，取最早的时间步）
    dataset = full_traffic_data.dataset
    if hasattr(dataset, 'traffic_x') and hasattr(dataset, 'traffic_y'):
        # PEMS04Dataset: 直接从内存数组切片，效率最高
        # traffic_x: [num_samples, seq_len, N, F], traffic_y: [num_samples, pred_len, N]
        actual_timesteps = min(sub_timesteps, len(dataset))
        sub_traffic = torch.FloatTensor(
            dataset.traffic_x[:actual_timesteps, :, :sub_nodes, :]
        ).to(device)
        sub_target = torch.FloatTensor(
            dataset.traffic_y[:actual_timesteps, :, :sub_nodes]
        ).to(device)
    else:
        # 回退方案: 迭代DataLoader收集数据（数据顺序为shuffle后的随机顺序）
        traffic_list, target_list = [], []
        collected = 0
        for batch in full_traffic_data:
            # 解包（兼容有/无天气数据两种格式）
            if len(batch) == 3:
                traffic, _, target = batch  # 预训练阶段忽略天气数据
            else:
                traffic, target = batch
            traffic_list.append(traffic)
            target_list.append(target)
            collected += traffic.size(0)
            if collected >= sub_timesteps:
                break
        # 拼接并截取子集: [sub_timesteps, seq_len, sub_nodes, input_dim]
        sub_traffic = torch.cat(traffic_list, dim=0)[:sub_timesteps, :, :sub_nodes, :].to(device)
        sub_target = torch.cat(target_list, dim=0)[:sub_timesteps, :, :sub_nodes].to(device)

    print(f"  [课程学习] 子集数据: traffic={sub_traffic.shape}, "
          f"target={sub_target.shape}")

    # ================================================================
    # 3. 构建子图邻接矩阵
    # ================================================================
    # 从全量邻接矩阵中提取左上角 sub_nodes x sub_nodes 子块
    sub_adj = adj_static[:sub_nodes, :sub_nodes].clone().to(device)

    # ================================================================
    # 4. 创建临时的小规模RLTDAGCN模型
    # ================================================================
    # 深拷贝配置，避免修改原始config
    sub_config = copy.deepcopy(config)
    sub_config.num_nodes = sub_nodes        # 子集节点数
    sub_config.use_multimodal = False       # 预训练阶段不使用天气数据，简化训练
    sub_config.use_topology = getattr(config, 'use_topology', True)  # 保持拓扑设置
    sub_config.use_rl = True                # 必须开启RL（预训练目标）

    sub_model = RLTDAGCN(sub_config, sub_adj).to(device)
    # 应用RL新参数到临时模型的控制器（动作空间收窄、奖励标准化等）
    sub_model.rl_controller.apply_config(config)

    # ================================================================
    # 5. 在子集上交替训练（主模型 + RL控制器）
    # ================================================================
    # 主模型优化器: 训练编码器、图生成器、空间精炼器等
    # 目的: 为RL控制器提供有意义的预测环境（非随机输出）
    sub_optimizer = torch.optim.Adam(
        sub_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    # RL控制器优化器: 仅更新PPO控制器参数
    rl_optimizer = torch.optim.Adam(
        sub_model.rl_controller.parameters(), lr=config.rl_lr
    )

    # 创建子集 DataLoader
    sub_dataset = TensorDataset(sub_traffic, sub_target)
    sub_batch_size = min(config.batch_size, sub_traffic.size(0))
    sub_loader = DataLoader(
        sub_dataset, batch_size=sub_batch_size, shuffle=True, drop_last=False
    )

    # --- 奖励函数参数（与 run_experiments.py 改进版保持一致）---
    reward_type = getattr(config, 'rl_reward_type', 'absolute')          # 奖励类型
    baseline_alpha = getattr(config, 'rl_reward_baseline_alpha', 0.9)    # EMA平滑系数
    stability_weight = getattr(config, 'rl_stability_weight', 0.05)      # 稳定性权重(absolute模式)
    diversity_bonus = getattr(config, 'rl_diversity_bonus', 0.0)         # 图多样性奖励系数
    oscillation_penalty = getattr(config, 'rl_oscillation_penalty', 0.0)  # 图震荡惩罚系数
    rollout_steps = getattr(config, 'rl_rollout_steps', 10)              # PPO更新阈值
    rl_mini_batch = getattr(config, 'rl_mini_batch_size', 16)            # PPO mini-batch
    rl_ppo_epochs = getattr(config, 'rl_ppo_epochs', 10)                 # PPO更新轮数

    # 相对奖励的EMA基线（初始为0，随训练逐步收敛到真实误差水平）
    baseline_error = 0.0

    for epoch in range(pretrain_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for traffic, target in sub_loader:
            traffic = traffic.to(device)
            target = target.to(device)

            sub_optimizer.zero_grad()

            # 前向传播（collect_rl=True: 采样探索 + 收集RL经验）
            pred, info = sub_model(
                traffic, None, sub_adj,
                target=target, collect_rl=True
            )
            loss = info['total_loss']
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(sub_model.parameters(), config.grad_clip)
            sub_optimizer.step()

            # --- 收集RL经验到缓冲区 ---
            if info['rl_data'] is not None:
                rl_data = info['rl_data']
                pred_error = info['pred_loss'].item()
                # RL状态向量6维: [speed_mean, speed_std, flow_mean,
                #                 pred_error, graph_change, graph_density]
                graph_change = rl_data['state'][0, 4].item()    # 第5维: 图变化量 ||A_t - A_{t-1}||
                graph_density = rl_data['state'][0, 5].item()   # 第6维: 图密度(非对角线边权重均值)

                # === 奖励计算（与 run_experiments.py 保持一致）===
                if reward_type == 'relative':
                    # 相对奖励: 与EMA基线比较，关注"是否变好"而非"绝对好坏"
                    # 当 pred_error < baseline_error 时奖励为正(变好了)，反之为负
                    reward = baseline_error - pred_error
                    # 更新EMA基线: alpha越大，基线更新越慢(更稳定)
                    baseline_error = baseline_alpha * baseline_error \
                        + (1 - baseline_alpha) * pred_error
                else:
                    # 绝对奖励: 旧逻辑 r = -(MAE + lambda * graph_change)
                    reward = -(pred_error + stability_weight * graph_change)

                # 图多样性奖励: 鼓励探索不同图结构（密度适中的图获得正奖励）
                reward += diversity_bonus * graph_density
                # 图震荡惩罚: 防止图频繁大幅变化（graph_change越大惩罚越重）
                reward -= oscillation_penalty * graph_change

                # 存入经验缓冲区
                sub_model.rl_buffer.push(
                    state=rl_data['state'].squeeze(0),              # [6]
                    action_disc=rl_data['action_disc'].squeeze(0),  # scalar
                    action_cont=rl_data['action_cont'].squeeze(0),  # [1]
                    log_prob=rl_data['log_prob'].squeeze(0),        # scalar
                    reward=reward,
                    value=rl_data['value'].squeeze(0),              # [1]
                    done=False,
                    pred_error=pred_error,
                )

            # --- PPO更新（经验积累到阈值后触发）---
            if len(sub_model.rl_buffer) > rollout_steps:
                rl_info = sub_model.rl_controller.ppo_update(
                    sub_model.rl_buffer, rl_optimizer,
                    mini_batch_size=rl_mini_batch,
                    ppo_epochs=rl_ppo_epochs,
                )
                sub_model.rl_buffer.clear()

            epoch_loss += loss.item()
            n_batches += 1

        # 每个epoch结束后重置图状态（prev_adj 恢复为静态图）
        sub_model.reset_graph_state()

        # 打印预训练进度（每5轮或第1轮打印）
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  [课程学习] Epoch {epoch+1}/{pretrain_epochs} "
                  f"Loss:{avg_loss:.4f} baseline_error:{baseline_error:.4f}")

    # ================================================================
    # 6. 将RL控制器权重复制回原始模型
    # ================================================================
    # RL控制器是节点数无关的MLP: state_dim=6, hidden_dim=64
    # 无论子图是50节点还是307节点，RL控制器的网络结构和权重shape完全一致
    # 因此可直接将小图上训练好的策略迁移到大图
    model.rl_controller.load_state_dict(sub_model.rl_controller.state_dict())
    print(f"  [课程学习] RL控制器权重已从子图({sub_nodes}节点)迁移到"
          f"全图({full_nodes}节点)")

    return model
