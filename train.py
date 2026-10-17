"""
RL-TDAGCN 训练主脚本
实现完整的训练流程，包括预热阶段和交替训练阶段

训练策略（交替训练）:
  阶段1 (epoch 0 ~ warmup_epochs-1): 预热阶段
    - 使用静态邻接矩阵
    - 只训练主模型（编码器+动态图+输出层）
    - RL控制器不参与

  阶段2 (epoch warmup_epochs ~): 交替训练
    - 每个epoch: 主模型正常训练，同时收集RL经验
    - 每 alternate_interval 个epoch: 用收集的经验更新RL控制器
    - 主模型使用RL优化后的动态图
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

# 将项目根目录加入系统路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from configs.default import Config
from models import RLTDAGCN
from utils import create_data_loaders, masked_mae, compute_all_metrics, EarlyStopping
from utils.gaode_loader import create_gaode_data_loaders


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RL-TDAGCN 训练脚本')

    # 数据参数
    parser.add_argument('--dataset', type=str,
                        default='../DAGGCN_ACMoE_RL/data/PEMS04',
                        help='数据集目录路径')
    parser.add_argument('--epochs', type=int, default=150,
                        help='训练轮数 (v5: 100->150, 给复杂变体更多收敛时间)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='主模型学习率')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='隐藏层维度')

    # 功能开关（使用action='store_true/false'正确处理布尔参数）
    parser.add_argument('--use_rl', action='store_true', default=True,
                        help='是否启用RL图控制器（默认开启，加 --no-use-rl 关闭）')
    parser.add_argument('--no-use_rl', dest='use_rl', action='store_false',
                        help='关闭RL图控制器')
    parser.add_argument('--use_topology', action='store_true', default=True,
                        help='是否启用拓扑约束（默认开启，加 --no-use_topology 关闭）')
    parser.add_argument('--no-use_topology', dest='use_topology', action='store_false',
                        help='关闭拓扑约束')
    parser.add_argument('--use_multimodal', action='store_true', default=False,
                        help='是否启用多模态融合')
    parser.add_argument('--no-use_multimodal', dest='use_multimodal', action='store_false',
                        help='关闭多模态融合')

    # 设备和保存
    parser.add_argument('--device', type=str, default='cpu',
                        help='训练设备')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='实验名称')

    # [v4新增] 断点续训
    parser.add_argument('--resume', action='store_true',
                        help='[v4] 断点续训: 从上次中断处继续训练')

    return parser.parse_args()


def evaluate(model, loader, adj_static, device, scaler_stats, has_weather):
    """
    在验证/测试集上评估模型

    Args:
        model: 模型实例
        loader: DataLoader
        adj_static: [N, N] 静态邻接矩阵
        device: 设备
        scaler_stats: 标准化统计信息（用于反标准化）
        has_weather: 是否包含天气数据

    Returns:
        dict: MAE, RMSE, SMAPE 指标
    """
    model.eval()
    # 评估前重置图状态为静态图
    if hasattr(model, 'reset_graph_state'):
        model.reset_graph_state()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if has_weather:
                traffic, weather, target = batch
                weather = weather.to(device)
            else:
                traffic, target = batch
                weather = None

            traffic = traffic.to(device)
            target = target.to(device)

            # 推理预测
            pred = model.predict(traffic, weather, adj_static)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # [num_samples, pred_len, N]
    all_targets = np.concatenate(all_targets, axis=0)

    # 反标准化: speed是第3个特征（index=2）
    speed_std = scaler_stats['std'][0, :, 2:3]  # [1, N, 1]
    speed_mean = scaler_stats['mean'][0, :, 2:3]

    all_preds_denorm = all_preds * speed_std.flatten() + speed_mean.flatten()
    all_targets_denorm = all_targets * speed_std.flatten() + speed_mean.flatten()

    # 计算所有指标
    metrics = compute_all_metrics(all_preds_denorm, all_targets_denorm)
    return metrics


def compute_rl_reward(pred, target, current_adj, prev_adj, stability_weight):
    """
    计算RL奖励 = -(MAE + lambda * graph_change)

    目标: 预测准确 + 图结构稳定

    Args:
        pred: [B, pred_len, N] 预测值（标准化）
        target: [B, pred_len, N] 真实值（标准化）
        current_adj: [N, N] 当前动态图
        prev_adj: [N, N] 或 None 上一时刻动态图
        stability_weight: 图稳定性惩罚权重

    Returns:
        reward: float RL奖励值
    """
    # 预测MAE
    mae = masked_mae(pred, target).item()

    # 图变化量（Frobenius范数）
    if prev_adj is not None:
        graph_change = torch.norm(current_adj - prev_adj, p='fro').item()
    else:
        graph_change = 0.0

    # 奖励 = 负的（MAE + 稳定性惩罚）
    reward = -(mae + stability_weight * graph_change)
    return reward


def train_one_epoch(model, train_loader, optimizer, adj_static, device,
                    epoch, config, has_weather, collect_rl=False, scaler=None):
    """
    训练一个epoch (v4: 支持AMP混合精度)

    Args:
        model: 模型实例
        train_loader: 训练DataLoader
        optimizer: 主模型优化器
        adj_static: [N, N] 静态邻接矩阵
        device: 设备
        epoch: 当前epoch编号
        config: 配置对象
        has_weather: 是否有天气数据
        collect_rl: 是否收集RL经验
        scaler: [v4新增] AMP GradScaler, None则不用混合精度

    Returns:
        avg_loss: 平均训练损失
    """
    model.train()
    # 每个epoch开始时重置图状态
    if hasattr(model, 'reset_graph_state'):
        model.reset_graph_state()

    # [v4新增] AMP混合精度: CUDA设备自动启用
    use_amp = scaler is not None and device.type == 'cuda'

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # 解包数据（处理有无天气两种情况）
        if has_weather:
            traffic, weather, target = batch
            weather = weather.to(device, non_blocking=True)
        else:
            traffic, target = batch
            weather = None

        traffic = traffic.to(device, non_blocking=True)    # [B, T, N, F]
        target = target.to(device, non_blocking=True)      # [B, pred_len, N]

        # [v4新增] AMP: 前向传播使用autocast自动混合精度
        if use_amp:
            with torch.cuda.amp.autocast():
                pred, info = model(traffic, weather, adj_static,
                                    target=target, collect_rl=collect_rl)
                total_loss_val = info['total_loss']
        else:
            # 前向传播（模型内部计算损失）
            pred, info = model(traffic, weather, adj_static,
                                target=target, collect_rl=collect_rl)
            total_loss_val = info['total_loss']

        # 反向传播 [v4: AMP使用scaler缩放梯度]
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(total_loss_val).backward()
            scaler.unscale_(optimizer)
        else:
            total_loss_val.backward()

        # 梯度裁剪（主模型参数）
        clip_params = list(model.encoder.parameters()) + \
                     list(model.graph_generator.parameters()) + \
                     list(model.spatial_refiner.parameters()) + \
                     list(model.output_layer.parameters())
        if hasattr(model, 'topo_constraint') and model.topo_constraint is not None:
            clip_params += list(model.topo_constraint.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=config.grad_clip)

        # [v4: AMP使用scaler.step]
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += total_loss_val.item()
        num_batches += 1

        # 收集RL经验（如果启用RL且在收集阶段）
        if collect_rl and model.rl_controller is not None and model.rl_buffer is not None:
            rl_data = info.get('rl_data')
            if rl_data is not None and rl_data['log_prob'] is not None:
                # 计算RL奖励
                reward = compute_rl_reward(
                    pred.detach(), target.detach(),
                    model.prev_adj,
                    info.get('A_candidate'),
                    config.rl_stability_weight
                )

                # 存入经验缓冲区
                model.rl_buffer.push(
                    state=rl_data['state'],
                    action_disc=rl_data['action_disc'],
                    action_cont=rl_data['action_cont'],
                    log_prob=rl_data['log_prob'],
                    reward=reward,
                    value=rl_data['value'],
                    done=torch.tensor(0.0, device=device),
                )

        # 日志打印
        if batch_idx % config.log_interval == 0:
            pred_loss = info.get('pred_loss', torch.tensor(0.0)).item()
            topo_loss = info.get('topo_loss', torch.tensor(0.0)).item()
            action_desc = info.get('action_desc', '')
            extra = f' | 动作: {action_desc}' if action_desc and collect_rl else ''
            print(f'    [Batch {batch_idx}/{len(train_loader)}] '
                  f'Loss: {total_loss_val.item():.4f} '
                  f'Pred: {pred_loss:.4f} '
                  f'Topo: {topo_loss:.4f}'
                  f'{extra}')

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    """主训练流程"""
    args = parse_args()

    # ========== 1. 配置初始化 ==========
    config = Config()
    # 用命令行参数覆盖默认配置
    config.data_dir = args.dataset
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.hidden_dim = args.hidden_dim
    config.use_rl = args.use_rl
    config.use_topology = args.use_topology
    config.use_multimodal = args.use_multimodal
    config.device = args.device
    config.save_dir = args.save_dir

    device = torch.device(config.device)

    # 随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 实验保存目录
    exp_save_dir = os.path.join(config.save_dir, args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)

    print('=' * 60)
    print(f'RL-TDAGCN 训练')
    print(f'实验名称: {args.exp_name}')
    print(f'保存目录: {exp_save_dir}')
    print(f'设备: {device}')
    print(f'配置: epochs={config.epochs}, batch_size={config.batch_size}, '
          f'lr={config.lr}, hidden_dim={config.hidden_dim}')
    print(f'RL={config.use_rl}, Topology={config.use_topology}, '
          f'Multimodal={config.use_multimodal}')
    print('=' * 60)

    # ========== 2. 数据加载 ==========
    print('\n[数据加载] ...')

    # 自动检测数据集类型: 如果目录下有 traffic.npy 则用高德加载器, 否则用 PEMS 加载器
    gaode_traffic_path = os.path.join(config.data_dir, 'traffic.npy')
    if os.path.exists(gaode_traffic_path):
        print(f'[数据加载] 检测到 traffic.npy, 使用高德数据加载器')
        train_loader, val_loader, test_loader, adj_mx, scaler_stats = \
            create_gaode_data_loaders(config.data_dir, config)
    else:
        print(f'[数据加载] 使用 PEMS 数据加载器')
        train_loader, val_loader, test_loader, adj_mx, scaler_stats = \
            create_data_loaders(config.data_dir, config)

    # 检测数据集是否包含天气数据
    weather_path = os.path.join(config.data_dir, 'weather_aligned.npy')
    has_weather = config.use_multimodal and os.path.exists(weather_path)
    if has_weather:
        print(f'[数据加载] 检测到天气数据，启用多模态模式')
    else:
        print(f'[数据加载] 无天气数据，使用纯交通模式')

    # 静态邻接矩阵转tensor
    adj_static = torch.FloatTensor(adj_mx).to(device)
    print(f'[数据加载] 邻接矩阵: {adj_static.shape}')

    # ========== 3. 模型初始化 ==========
    print('\n[模型初始化] ...')
    model = RLTDAGCN(config, adj_static).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[模型初始化] 总参数量: {total_params:,}')
    print(f'[模型初始化] 可训练参数量: {trainable_params:,}')

    # 两个优化器: 主模型 + RL控制器
    # 主模型优化器: 编码器 + 动态图生成器 + 空间精炼 + 输出层 (+ 拓扑约束如果启用)
    main_params = list(model.encoder.parameters()) + \
                 list(model.graph_generator.parameters()) + \
                 list(model.spatial_refiner.parameters()) + \
                 list(model.output_layer.parameters())
    if hasattr(model, 'topo_constraint') and model.topo_constraint is not None:
        main_params += list(model.topo_constraint.parameters())

    main_optimizer = torch.optim.Adam(
        main_params,
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    rl_optimizer = None
    if config.use_rl and hasattr(model, 'rl_controller') and model.rl_controller is not None:
        rl_optimizer = torch.optim.Adam(
            model.rl_controller.parameters(),
            lr=config.rl_lr
        )

    # 早停机制
    early_stopping = EarlyStopping(
        mode='min',
        patience=config.patience,
        verbose=True
    )

    # ========== [v4新增] AMP混合精度初始化 ==========
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        print('[v4] AMP混合精度训练已启用 (CUDA设备自动检测)')

    # ========== [v4新增] 断点续训: checkpoint路径 ==========
    ckpt_path = os.path.join(exp_save_dir, 'checkpoint_latest.pt')
    start_epoch = 0

    if args.resume and os.path.exists(ckpt_path):
        print(f'[v4] 检测到checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        main_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scaler is not None and ckpt.get('scaler_state_dict'):
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        # 恢复早停状态
        es = ckpt.get('early_stopping_state', {})
        early_stopping.counter = es.get('counter', 0)
        early_stopping.best_score = es.get('best_score', None)
        early_stopping.best_epoch = es.get('best_epoch', 0)
        early_stopping.best_model_state = es.get('best_model_state', None)
        # 恢复训练日志
        training_log = ckpt.get('training_log', [])
        print(f'[v4] 从 epoch {start_epoch} 恢复训练 (已完成 {start_epoch} epochs)')
    else:
        if args.resume:
            print(f'[v4] 未找到checkpoint, 从头开始训练')
        training_log = []

    # ========== 4. 训练循环 ==========
    print('\n[训练开始] ...')
    warmup_epochs = config.rl_warmup_epochs if config.use_rl else 0
    alternate_interval = config.rl_alternate_interval

    start_time = time.time()

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # 判断当前阶段
        is_warmup = epoch < warmup_epochs
        collect_rl = (not is_warmup) and config.use_rl

        # 训练
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=main_optimizer,
            adj_static=adj_static,
            device=device,
            epoch=epoch,
            config=config,
            has_weather=has_weather,
            collect_rl=collect_rl,
            scaler=scaler,  # [v4新增] AMP GradScaler
        )

        # 验证
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            adj_static=adj_static,
            device=device,
            scaler_stats=scaler_stats,
            has_weather=has_weather,
        )

        # 日志
        phase_str = '预热' if is_warmup else '交替'
        epoch_time = time.time() - epoch_start
        print(f'[Epoch {epoch + 1}/{config.epochs}] '
              f'[{phase_str}] '
              f'Train Loss: {train_loss:.4f} | '
              f'Val MAE: {val_metrics["MAE"]:.2f} '
              f'RMSE: {val_metrics["RMSE"]:.2f} '
              f'SMAPE: {val_metrics["SMAPE"] * 100:.1f}% | '
              f'Time: {epoch_time:.1f}s')

        # 早停检查（使用验证集MAE）
        should_stop = early_stopping(val_metrics['MAE'], epoch, model)

        # RL更新（交替训练）
        # v3改进: 增加最小buffer长度检查, 防止在太少经验上更新PPO
        if (not is_warmup) and config.use_rl and rl_optimizer is not None:
            rl_epoch_from_warmup = epoch - warmup_epochs
            if rl_epoch_from_warmup > 0 and rl_epoch_from_warmup % alternate_interval == 0:
                # [v3新增] 确保buffer中有足够经验再更新
                min_buffer_size = max(config.rl_mini_batch_size, 10)
                if len(model.rl_buffer) >= min_buffer_size:
                    # 使用收集的经验更新RL控制器
                    rl_info = model.rl_controller.ppo_update(
                        buffer=model.rl_buffer,
                        optimizer=rl_optimizer,
                        mini_batch_size=config.rl_mini_batch_size,
                        ppo_epochs=config.rl_ppo_epochs,
                    )
                    # 清空经验缓冲区
                    model.rl_buffer.clear()
                    print(f'  [RL] policy_loss: {rl_info["policy_loss"]:.4f}, '
                          f'value_loss: {rl_info["value_loss"]:.4f}, '
                          f'entropy: {rl_info["entropy"]:.4f}')
                elif len(model.rl_buffer) > 0:
                    print(f'  [RL] buffer太小({len(model.rl_buffer)}<{min_buffer_size}), 跳过更新')
                else:
                    print(f'  [RL] 缓冲区为空，跳过更新')

        # 记录日志
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_mae': float(val_metrics['MAE']),
            'val_rmse': float(val_metrics['RMSE']),
            'val_smape': float(val_metrics['SMAPE']),
            'time': float(epoch_time),
            'phase': phase_str,
        }
        training_log.append(log_entry)

        # 早停判断
        if should_stop:
            print(f'\n  [EarlyStop] 触发早停! 最佳 epoch: {early_stopping.best_epoch + 1}')
            break

        # === [v4新增] 每5个epoch保存checkpoint ===
        if (epoch + 1) % 5 == 0:
            es_state = {
                'counter': early_stopping.counter,
                'best_score': float(early_stopping.best_score) if early_stopping.best_score is not None else None,
                'best_epoch': early_stopping.best_epoch,
                'best_model_state': early_stopping.best_model_state,
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': main_optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'early_stopping_state': es_state,
                'training_log': training_log,
            }, ckpt_path)
            print(f'  [v4 Checkpoint] 已保存 epoch {epoch + 1} -> {ckpt_path}')

    total_time = time.time() - start_time
    print(f'\n[训练完成] 总耗时: {total_time:.1f}s ({total_time / 60:.1f}min)')

    # [v4新增] 训练完成后删除checkpoint (节省磁盘空间)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f'[v4] 训练完成, 已清理checkpoint: {ckpt_path}')

    # ========== 5. 测试 ==========
    print('\n[测试阶段] 使用最佳模型在测试集上评估...')

    # 加载最佳模型
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        print('[测试阶段] 已加载早停保存的最佳模型')
    else:
        print('[测试阶段] 未找到最佳模型，使用当前模型')

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        adj_static=adj_static,
        device=device,
        scaler_stats=scaler_stats,
        has_weather=has_weather,
    )

    print(f'\n===== 测试结果 =====')
    print(f'  MAE:  {test_metrics["MAE"]:.4f}')
    print(f'  RMSE: {test_metrics["RMSE"]:.4f}')
    print(f'  SMAPE: {test_metrics["SMAPE"] * 100:.2f}%')
    print(f'=====================')

    # ========== 6. 结果保存 ==========
    # 保存最佳模型
    best_model_path = os.path.join(exp_save_dir, 'best_model.pt')
    if early_stopping.best_model_state is not None:
        torch.save({
            'model_state_dict': early_stopping.best_model_state,
            'config': {k: v for k, v in vars(config).items()
                       if not k.startswith('_')},
            'best_epoch': early_stopping.best_epoch + 1,
            'best_val_mae': float(early_stopping.best_score),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        }, best_model_path)
        print(f'\n[保存] 最佳模型: {best_model_path}')

    # 保存训练结果为JSON
    results = {
        'config': {k: v for k, v in vars(config).items()
                   if not k.startswith('_')},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'best_epoch': early_stopping.best_epoch + 1,
        'best_val_mae': float(early_stopping.best_score) if early_stopping.best_score is not None else None,
        'total_epochs': epoch + 1,
        'total_time_seconds': total_time,
        'training_log': training_log,
    }

    results_path = os.path.join(exp_save_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'[保存] 训练结果: {results_path}')

    # 保存训练日志（纯文本格式，方便查看）
    log_path = os.path.join(exp_save_dir, 'training.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'RL-TDAGCN 训练日志\n')
        f.write(f'实验名称: {args.exp_name}\n')
        f.write(f'设备: {device}\n')
        f.write(f'{"=" * 60}\n\n')
        for entry in training_log:
            f.write(
                f'[Epoch {entry["epoch"]}/{config.epochs}] '
                f'[{entry["phase"]}] '
                f'Train Loss: {entry["train_loss"]:.4f} | '
                f'Val MAE: {entry["val_mae"]:.2f} '
                f'RMSE: {entry["val_rmse"]:.2f} '
                f'SMAPE: {entry["val_smape"] * 100:.1f}% | '
                f'Time: {entry["time"]:.1f}s\n'
            )
        f.write(f'\n{"=" * 60}\n')
        f.write(f'测试结果:\n')
        f.write(f'  MAE:  {test_metrics["MAE"]:.4f}\n')
        f.write(f'  RMSE: {test_metrics["RMSE"]:.4f}\n')
        f.write(f'  SMAPE: {test_metrics["SMAPE"] * 100:.2f}%\n')
        f.write(f'总耗时: {total_time:.1f}s ({total_time / 60:.1f}min)\n')
    print(f'[保存] 训练日志: {log_path}')

    print('\n[完成] 训练流程结束。')
    return test_metrics


if __name__ == '__main__':
    main()
