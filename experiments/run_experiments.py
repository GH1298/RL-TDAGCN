"""
5组实验脚本 (v4: 断点续训 + AMP混合精度)
自动运行所有实验（基础模型对比 + 4组消融实验）并保存结果为JSON。

v4 改进:
  1. 断点续训: 每个variant训练中每5个epoch保存checkpoint, 崩溃可恢复
  2. 实验级跳过: 已完成的variant自动跳过, 不重复训练
  3. AMP混合精度: torch.cuda.amp, 约1.5x加速, 精度无损
  4. DataLoader优化: num_workers=4 + pin_memory (在dataloader.py中)

实验设计:
  实验1: 基础模型对比 (LSTM, STGCN, TGCN, DCRNN, GWN, RL-TDAGCN)
  实验2: RL 有效性消融 (DAGGCN vs DAGGCN+RL)
  实验3: Topology 有效性消融 (DAGGCN vs DAGGCN+Topo)
  实验4: 联合效果 (Base / +RL / +Topo / +RL+Topo)
  实验5: 多模态消融 (Traffic only vs +Weather)

用法:
  python -m experiments.run_experiments --experiment 1 --quick
  python -m experiments.run_experiments --experiment all
  python -m experiments.run_experiments --experiment 2 --epochs 50
  python -m experiments.run_experiments --experiment 4 --resume  # 断点续训
"""
import argparse
import json
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 将项目根目录加入路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_data_loaders, compute_all_metrics, masked_mae, EarlyStopping
from models.rlt_dagcn import RLTDAGCN
from models.baselines import DCRNNBaseline, GraphWaveNet, STGCN, TGCN, LSTMModel
from configs.default import Config

# [v5修复] curriculum_pretrain 改为懒加载，避免文件缺失时所有实验崩溃
# 课程学习在v5中已关闭(rl_curriculum_enabled=False)，此模块仅在需要时才导入
try:
    from experiments.curriculum_pretrain import pretrain_rl_controller
    _HAS_CURRICULUM = True
except ImportError:
    pretrain_rl_controller = None
    _HAS_CURRICULUM = False


# ============================================================
# v4: 断点续训工具函数
# ============================================================

def get_checkpoint_path(save_dir, exp_id, variant_name):
    """
    获取variant的checkpoint文件路径

    Args:
        save_dir: 结果保存目录
        exp_id: 实验编号
        variant_name: 模型/变体名称

    Returns:
        checkpoint文件路径
    """
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    # 文件名: exp4_Base_latest.pt
    safe_name = variant_name.replace('+', '_').replace(' ', '_')
    return os.path.join(ckpt_dir, f'exp{exp_id}_{safe_name}_latest.pt')


def save_checkpoint(path, model, optimizer, epoch, early_stop_state,
                    best_metrics, training_log, config_dict):
    """
    保存训练checkpoint

    保存内容:
        - model_state_dict: 模型权重
        - optimizer_state_dict: 优化器状态
        - epoch: 当前epoch编号 (0-based)
        - early_stop_state: 早停机制的状态 (best_score, best_epoch, best_model_state, counter)
        - best_metrics: 最佳验证指标
        - training_log: 训练日志列表
        - config_dict: 配置字典

    Args:
        path: 保存路径
        model: 模型实例
        optimizer: 优化器
        epoch: 当前epoch (0-based)
        early_stop_state: 早停状态字典
        best_metrics: 最佳指标
        training_log: 训练日志
        config_dict: 配置
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'early_stop_state': early_stop_state,
        'best_metrics': best_metrics,
        'training_log': training_log,
        'config_dict': config_dict,
    }, path)


def load_checkpoint(path, model, optimizer, device):
    """
    加载checkpoint并恢复训练状态

    Args:
        path: checkpoint文件路径
        model: 模型实例 (权重会被覆盖)
        optimizer: 优化器 (状态会被覆盖)
        device: 设备

    Returns:
        ckpt: checkpoint字典, 包含 epoch, early_stop_state, best_metrics, training_log
        None: 如果文件不存在
    """
    if not os.path.exists(path):
        return None

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and ckpt['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    print(f"  [Resume] 从 epoch {ckpt['epoch'] + 1} 恢复训练 "
          f"(已训练 {ckpt['epoch'] + 1} epochs)")
    return ckpt


def load_partial_results(save_dir, exp_id):
    """
    加载已完成的variant结果 (实验级断点)

    如果 exp{N}_results.json 已存在, 说明该实验所有variant都已完成,
    返回完整结果字典。否则返回None。

    Args:
        save_dir: 结果保存目录
        exp_id: 实验编号

    Returns:
        results dict 或 None
    """
    result_file = os.path.join(save_dir, f'exp{exp_id}_results.json')
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_partial_results(save_dir, exp_id, results):
    """
    保存当前已完成的variant结果

    每完成一个variant就保存一次, 这样即使后续variant崩溃,
    已完成的也不需要重跑。

    Args:
        save_dir: 结果保存目录
        exp_id: 实验编号
        results: 当前已完成的结果字典
    """
    result_file = os.path.join(save_dir, f'exp{exp_id}_results.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ============================================================
# 训练+评估函数 (v4: 添加AMP + checkpoint)
# ============================================================

def train_and_evaluate(model, train_loader, val_loader, test_loader,
                       adj_static, scaler_stats, config, device,
                       model_name, save_dir, has_weather=False,
                       exp_id=0, resume=False):
    """
    通用训练+评估函数 (v4: AMP混合精度 + 断点续训)

    对于RL-TDAGCN模型: 使用交替训练（RL在预热后按间隔更新）
    对于基线模型: 标准训练（纯MAE损失）

    v4新增:
        - AMP混合精度: 使用torch.cuda.amp加速训练 (~1.5x)
        - 断点续训: 每5个epoch保存checkpoint, 支持恢复
        - 实验级跳过: 已完成的variant自动跳过

    Args:
        model:        模型实例
        train_loader: 训练集 DataLoader
        val_loader:   验证集 DataLoader
        test_loader:  测试集 DataLoader
        adj_static:   [N, N] 静态邻接矩阵 (torch.Tensor, 已在device上)
        scaler_stats: 标准化统计量 (用于反标准化)
        config:       配置对象
        device:       训练设备
        model_name:   模型名称 (用于日志)
        save_dir:     结果保存目录
        has_weather:  数据集是否包含天气数据
        exp_id:       实验编号 (用于checkpoint命名)
        resume:       是否从checkpoint恢复

    Returns:
        test_metrics: dict, 测试集指标 (MAE, RMSE, SMAPE)
    """
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # 早停机制
    early_stop = EarlyStopping(mode='min', patience=config.patience, verbose=True)

    # === [v4新增] AMP混合精度初始化 ===
    # GradScaler用于缩放loss防止梯度下溢
    # autocast在前向传播中自动选择半精度(fp16)或全精度(fp32)计算
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # === [P1新增] 应用RL新参数到控制器 ===
    if isinstance(model, RLTDAGCN) and model.use_rl:
        model.rl_controller.apply_config(config)

    # === [P1新增] 课程学习 ===
    # [v5修复] 增加 _HAS_CURRICULUM 检查，文件缺失时跳过而非崩溃
    if (isinstance(model, RLTDAGCN) and model.use_rl and
            getattr(config, 'rl_curriculum_enabled', False)):
        if _HAS_CURRICULUM and pretrain_rl_controller is not None:
            print(f"  [{model_name}] 开始课程学习预训练...")
            model = pretrain_rl_controller(
                model, train_loader, adj_static, config, device
            )
        else:
            print(f"  [{model_name}] 警告: curriculum_pretrain模块未找到，跳过课程学习")
            print(f"           (v5中课程学习已关闭，不影响实验)")

    # === [P1新增] 相对奖励的EMA基线 ===
    baseline_error = 0.0

    best_metrics = None
    start_time = time.time()
    training_log = []

    # === [v4新增] 断点续训: 尝试加载checkpoint ===
    start_epoch = 0
    ckpt_path = get_checkpoint_path(save_dir, exp_id, model_name)

    if resume and os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, model, optimizer, device)
        if ckpt is not None:
            start_epoch = ckpt['epoch'] + 1  # 从下一个epoch继续
            training_log = ckpt.get('training_log', [])
            best_metrics = ckpt.get('best_metrics', None)

            # 恢复早停状态
            es_state = ckpt.get('early_stop_state', {})
            if es_state:
                early_stop.best_score = es_state.get('best_score', None)
                early_stop.best_epoch = es_state.get('best_epoch', 0)
                early_stop.counter = es_state.get('counter', 0)
                if es_state.get('best_model_state'):
                    early_stop.best_model_state = es_state.get('best_model_state')

            # 恢复EMA基线
            baseline_error = es_state.get('baseline_error', 0.0)

            print(f"  [{model_name}] 恢复到 epoch {start_epoch}, 继续训练...")
        else:
            print(f"  [{model_name}] 无有效checkpoint, 从头开始训练")
    else:
        if resume:
            print(f"  [{model_name}] 未找到checkpoint ({ckpt_path}), 从头开始训练")
        else:
            print(f"  [{model_name}] 从头开始训练 (无断点续训)")

    # === 训练循环 ===
    for epoch in range(start_epoch, config.epochs):
        # ==========================================
        # [v5新增] 判断当前epoch是否为PPO更新epoch
        # 如果是且rl_freeze_main_model=True, 则跳过主模型梯度更新
        # 原因: PPO更新改变图策略后, 主模型用新图做forward+backward会导致特征失配
        # ==========================================
        is_ppo_update_epoch = (
            isinstance(model, RLTDAGCN) and model.use_rl and
            epoch >= config.rl_warmup_epochs and
            (epoch - config.rl_warmup_epochs) % config.rl_alternate_interval == 0 and
            len(getattr(model, 'rl_buffer', None) or []) > config.rl_rollout_steps
        )
        freeze_main = (
            is_ppo_update_epoch and
            getattr(config, 'rl_freeze_main_model', False)
        )
        if freeze_main:
            print(f"  [v5 Freeze] Epoch {epoch+1}: PPO更新epoch, 冻结主模型梯度")

        # ==========================================
        # 训练阶段 (v4: AMP混合精度, v5: 梯度冻结)
        # ==========================================
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # 解包数据（有/无天气两种情况）
            if has_weather:
                traffic, weather, target = batch
                weather = weather.to(device, non_blocking=True)
            else:
                traffic, target = batch
                weather = None

            traffic = traffic.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            if isinstance(model, RLTDAGCN):
                # === RL-TDAGCN: 完整模型训练 (v4: AMP) ===
                # 使用autocast自动混合精度, 前向传播中部分计算用fp16
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred, info = model(traffic, weather, adj_static,
                                       target=target,
                                       collect_rl=(epoch >= config.rl_warmup_epochs))
                    loss = info['total_loss']

                # 收集RL经验到缓冲区
                if (model.use_rl and info['rl_data'] is not None and
                    epoch >= config.rl_warmup_epochs):
                    rl_data = info['rl_data']
                    pred_error = info['pred_loss'].item()
                    graph_change = rl_data['state'][0, 4].item()
                    graph_density = rl_data['state'][0, 5].item()

                    # 奖励函数
                    reward_type = getattr(config, 'rl_reward_type', 'absolute')
                    if reward_type == 'relative':
                        reward = baseline_error - pred_error
                        baseline_alpha = getattr(config, 'rl_reward_baseline_alpha', 0.9)
                        baseline_error = baseline_alpha * baseline_error \
                            + (1 - baseline_alpha) * pred_error
                    else:
                        stability_weight = getattr(config, 'rl_stability_weight', 0.05)
                        reward = -(pred_error + stability_weight * graph_change)

                    diversity_bonus = getattr(config, 'rl_diversity_bonus', 0.0)
                    oscillation_penalty = getattr(config, 'rl_oscillation_penalty', 0.0)
                    reward += diversity_bonus * graph_density
                    reward -= oscillation_penalty * graph_change

                    model.rl_buffer.push(
                        state=rl_data['state'].squeeze(0),
                        action_disc=rl_data['action_disc'].squeeze(0),
                        action_cont=rl_data['action_cont'].squeeze(0),
                        log_prob=rl_data['log_prob'].squeeze(0),
                        reward=reward,
                        value=rl_data['value'].squeeze(0),
                        done=False,
                        pred_error=pred_error,
                    )

                # [v5新增] 梯度冻结: PPO更新epoch跳过主模型反向传播
                if freeze_main:
                    # 只记录loss用于日志, 不做反向传播
                    total_loss += loss.item()
                    n_batches += 1
                    continue

                # PPO更新 (注意: PPO更新在autocast之外, 保持全精度)
                if (model.use_rl and
                    epoch >= config.rl_warmup_epochs and
                    (epoch - config.rl_warmup_epochs) % config.rl_alternate_interval == 0 and
                    len(model.rl_buffer) > config.rl_rollout_steps):
                    rl_opt = torch.optim.Adam(model.rl_controller.parameters(), lr=config.rl_lr)
                    rl_info = model.rl_controller.ppo_update(
                        model.rl_buffer, rl_opt,
                        mini_batch_size=config.rl_mini_batch_size,
                        ppo_epochs=config.rl_ppo_epochs
                    )
                    model.rl_buffer.clear()
                    print(f"    [RL Update] policy_loss={rl_info['policy_loss']:.4f} "
                          f"value_loss={rl_info['value_loss']:.4f} "
                          f"entropy={rl_info['entropy']:.4f}")
            else:
                # === 基线模型: 标准MAE损失 (v4: AMP) ===
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred, _ = model(traffic, adj_static)
                    loss = masked_mae(pred, target)

            # === [v4] AMP反向传播 ===
            # scaler.scale: 放大loss防止fp16梯度下溢
            # scaler.step: 检查梯度是否溢出, 安全地调用optimizer.step
            # scaler.update: 动态调整缩放因子
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # [v5新增] 梯度冻结模式: PPO更新在batch循环外执行
        # 当freeze_main=True时, batch循环中跳过了PPO更新, 需要在此处补上
        if freeze_main and isinstance(model, RLTDAGCN) and model.use_rl:
            if len(model.rl_buffer) > config.rl_rollout_steps:
                rl_opt = torch.optim.Adam(model.rl_controller.parameters(), lr=config.rl_lr)
                rl_info = model.rl_controller.ppo_update(
                    model.rl_buffer, rl_opt,
                    mini_batch_size=config.rl_mini_batch_size,
                    ppo_epochs=config.rl_ppo_epochs
                )
                model.rl_buffer.clear()
                print(f"    [v5 RL Update (frozen)] policy_loss={rl_info['policy_loss']:.4f} "
                      f"value_loss={rl_info['value_loss']:.4f} "
                      f"entropy={rl_info['entropy']:.4f}")

        # ==========================================
        # 验证阶段
        # ==========================================
        val_metrics = evaluate(model, val_loader, adj_static, device, scaler_stats,
                               has_weather, is_rlt=isinstance(model, RLTDAGCN))
        val_mae = val_metrics['MAE']

        # 打印训练日志
        elapsed = time.time() - start_time
        print(f"  [{model_name}] Epoch {epoch+1}/{config.epochs} "
              f"Loss:{avg_loss:.4f} Val MAE:{val_mae:.4f} "
              f"[{elapsed:.1f}s]")

        # 记录日志
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': float(avg_loss),
            'val_mae': float(val_mae),
            'val_rmse': float(val_metrics['RMSE']),
            'val_smape': float(val_metrics['SMAPE']),
            'time': float(elapsed),
        })

        # 早停判断
        should_stop = early_stop(val_mae, epoch, model)

        # 更新最佳指标
        if early_stop.best_score is not None and \
           (best_metrics is None or val_mae < best_metrics.get('MAE', float('inf'))):
            best_metrics = val_metrics.copy()

        # 每个epoch结束后重置RL模型图状态
        if isinstance(model, RLTDAGCN) and model.use_rl:
            model.reset_graph_state()

        # === [v4新增] 每5个epoch保存checkpoint ===
        if (epoch + 1) % 5 == 0 or should_stop:
            early_stop_state = {
                'best_score': float(early_stop.best_score) if early_stop.best_score is not None else None,
                'best_epoch': early_stop.best_epoch,
                'counter': early_stop.counter,
                'best_model_state': early_stop.best_model_state,
                'baseline_error': baseline_error,
            }
            config_dict = {k: v for k, v in vars(config).items()
                          if not k.startswith('_') and not callable(v)}
            save_checkpoint(ckpt_path, model, optimizer, epoch,
                           early_stop_state, best_metrics,
                           training_log, config_dict)

        # 早停判断
        if should_stop:
            print(f"\n  [{model_name}] EarlyStop! 最佳 epoch: {early_stop.best_epoch + 1}")
            break

    # 恢复最佳模型权重
    if early_stop.best_model_state is not None:
        model.load_state_dict(early_stop.best_model_state)

    # ==========================================
    # 测试阶段
    # ==========================================
    test_metrics = evaluate(model, test_loader, adj_static, device, scaler_stats,
                            has_weather, is_rlt=isinstance(model, RLTDAGCN))
    total_time = time.time() - start_time
    print(f"  [{model_name}] Test: MAE={test_metrics['MAE']:.4f} "
          f"RMSE={test_metrics['RMSE']:.4f} "
          f"SMAPE={test_metrics['SMAPE']*100:.2f}% "
          f"[总耗时: {total_time:.1f}s]")

    # === [v5修改] 保留checkpoint + 保存最佳模型权重 ===
    # v4逻辑: 训练完成后删除checkpoint节省空间
    # v5修改: 保留checkpoint用于断点续训, 额外保存最佳模型权重用于推理/可视化
    # 1. 保留训练checkpoint (含优化器状态, 用于resume)
    if os.path.exists(ckpt_path):
        print(f"  [{model_name}] 训练完成, checkpoint已保留: {ckpt_path}")

    # 2. 保存最佳模型权重 (仅模型权重, 不含优化器, 体积更小, 用于推理)
    best_model_dir = os.path.join(save_dir, 'best_models')
    os.makedirs(best_model_dir, exist_ok=True)
    safe_name = model_name.replace('+', '_').replace(' ', '_')
    best_model_path = os.path.join(best_model_dir, f'exp{exp_id}_{safe_name}_best.pt')
    if early_stop.best_model_state is not None:
        torch.save({
            'model_state_dict': early_stop.best_model_state,
            'best_epoch': early_stop.best_epoch,
            'best_metrics': best_metrics,
            'model_name': model_name,
            'exp_id': exp_id,
        }, best_model_path)
        print(f"  [{model_name}] 最佳模型已保存: {best_model_path} "
              f"(epoch={early_stop.best_epoch + 1})")
    else:
        # 如果没有best_model_state (如基线模型), 保存当前模型权重
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_epoch': -1,
            'best_metrics': best_metrics,
            'model_name': model_name,
            'exp_id': exp_id,
        }, best_model_path)
        print(f"  [{model_name}] 模型权重已保存: {best_model_path}")

    return test_metrics


@torch.no_grad()
def evaluate(model, loader, adj_static, device, scaler_stats, has_weather, is_rlt=False):
    """
    评估函数

    在指定数据集上运行模型推理，收集所有预测和真实值后统一计算指标。

    Args:
        model:        模型实例
        loader:       DataLoader
        adj_static:   [N, N] 静态邻接矩阵
        device:       设备
        scaler_stats: 标准化统计量 (mean, std)
        has_weather:  是否包含天气数据
        is_rlt:       是否为RL-TDAGCN模型

    Returns:
        metrics: dict, 包含 MAE, RMSE, SMAPE
    """
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        # 解包数据
        if has_weather:
            traffic, weather, target = batch
            weather = weather.to(device, non_blocking=True)
        else:
            traffic, target = batch
            weather = None
        traffic = traffic.to(device, non_blocking=True)

        # 推理
        if is_rlt and hasattr(model, 'predict'):
            # RL-TDAGCN 使用确定性推理方法
            pred = model.predict(traffic, weather, adj_static)
        else:
            # 基线模型使用标准 forward
            pred, _ = model(traffic, adj_static)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    # 拼接所有batch的结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 反标准化: 只对速度特征 (index=2) 反标准化
    speed_std = scaler_stats['std'][0, :, 2:3].flatten()  # [N]
    speed_mean = scaler_stats['mean'][0, :, 2:3].flatten()  # [N]
    all_preds = all_preds * speed_std + speed_mean
    all_targets = all_targets * speed_std + speed_mean

    # 计算所有评估指标
    return compute_all_metrics(all_preds, all_targets)


# ======================================================================
# 实验配置定义
# ======================================================================
EXPERIMENT_CONFIGS = {
    1: {
        'name': '基础模型对比',
        'models': [
            ('LSTM', lambda c, adj: LSTMModel(c.input_dim, c.hidden_dim, c.pred_len)),
            ('STGCN', lambda c, adj: STGCN(c.input_dim, c.hidden_dim, c.pred_len)),
            ('TGCN', lambda c, adj: TGCN(c.input_dim, c.hidden_dim, c.pred_len)),
            ('DCRNN', lambda c, adj: DCRNNBaseline(c.input_dim, c.hidden_dim, c.pred_len)),
            ('GWN', lambda c, adj: GraphWaveNet(c.input_dim, c.hidden_dim, c.num_nodes, pred_len=c.pred_len)),
            ('RL-TDAGCN', lambda c, adj: RLTDAGCN(c, torch.FloatTensor(adj))),
        ]
    },
    2: {
        'name': 'RL有效性消融',
        'configs': [
            ('DAGGCN', {'use_rl': False, 'use_topology': False, 'use_multimodal': False}),
            ('DAGGCN+RL', {'use_rl': True, 'use_topology': False, 'use_multimodal': False}),
        ]
    },
    3: {
        'name': 'Topology有效性消融',
        'configs': [
            ('DAGGCN', {'use_rl': False, 'use_topology': False, 'use_multimodal': False}),
            ('DAGGCN+Topo', {'use_rl': False, 'use_topology': True, 'use_multimodal': False}),
        ]
    },
    4: {
        'name': '联合效果',
        'configs': [
            ('Base', {'use_rl': False, 'use_topology': False, 'use_multimodal': False}),
            ('+RL', {'use_rl': True, 'use_topology': False, 'use_multimodal': False}),
            ('+Topo', {'use_rl': False, 'use_topology': True, 'use_multimodal': False}),
            ('+RL+Topo', {'use_rl': True, 'use_topology': True, 'use_multimodal': False}),
        ]
    },
    5: {
        'name': '多模态消融',
        'configs': [
            ('Traffic', {'use_rl': True, 'use_topology': True, 'use_multimodal': False}),
            ('+Weather', {'use_rl': True, 'use_topology': True, 'use_multimodal': True}),
        ]
    },
}


def run_single_experiment(exp_id, exp_config, train_loader, val_loader, test_loader,
                          adj_static, adj_mx, scaler_stats, device, args, resume=False):
    """
    运行单组实验 (v4: 支持断点续训)

    v4新增:
        - 如果 resume=True 且 exp{N}_results.json 已存在, 跳过整个实验
        - 每完成一个variant就保存部分结果, 崩溃后已完成的不需重跑

    Args:
        exp_id:       实验编号 (1-5)
        exp_config:   实验配置字典
        train_loader: 训练集 DataLoader
        val_loader:   验证集 DataLoader
        test_loader:  测试集 DataLoader
        adj_static:   [N, N] 静态邻接矩阵 (torch.Tensor)
        adj_mx:       [N, N] 静态邻接矩阵 (numpy, 用于模型构造)
        scaler_stats: 标准化统计量
        device:       训练设备
        args:         命令行参数
        resume:       是否断点续训

    Returns:
        results: dict, 每个模型/变体的测试指标
    """
    # === [v4新增] 实验级断点: 检查是否已有完整结果 ===
    if resume:
        existing = load_partial_results(args.save_dir, exp_id)
        if existing is not None:
            print(f"  [Skip] 实验{exp_id} 已有完整结果 ({len(existing)} 个variant), 跳过")
            return existing

    results = {}

    # === [v4新增] 断点续训: 加载已完成的部分结果 ===
    if resume:
        existing = load_partial_results(args.save_dir, exp_id)
        if existing is not None:
            results = existing
            print(f"  [Resume] 已完成 {len(results)} 个variant: {list(results.keys())}")

    # 判断数据集是否有天气数据
    weather_path = os.path.join(args.dataset, 'weather_aligned.npy')
    has_weather = os.path.exists(weather_path)

    if 'models' in exp_config:
        # === 实验1: 基线模型对比 ===
        for model_name, model_fn in exp_config['models']:
            # [v4] 跳过已完成的variant
            if model_name in results:
                print(f"\n--- {model_name} --- [已完成, 跳过]")
                continue

            print(f"\n--- {model_name} ---")

            # 构建基础配置
            config = Config()
            config.epochs = args.epochs
            config.batch_size = args.batch_size
            config.hidden_dim = args.hidden_dim
            config.device = args.device

            # 构造模型并移到设备
            model = model_fn(config, adj_mx).to(device)

            # 训练+评估 (v4: 传入exp_id和resume)
            metrics = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                adj_static, scaler_stats, config, device,
                model_name, args.save_dir, has_weather,
                exp_id=exp_id, resume=resume
            )
            results[model_name] = metrics

            # [v4] 每完成一个variant就保存
            save_partial_results(args.save_dir, exp_id, results)
            print(f"  [保存] {model_name} 结果已保存")
    else:
        # === 实验2-5: 消融实验 ===
        for variant_name, variant_config in exp_config['configs']:
            # [v4] 跳过已完成的variant
            if variant_name in results:
                print(f"\n--- {variant_name} --- [已完成, 跳过]")
                continue

            print(f"\n--- {variant_name} ---")

            # 为每个变体创建独立配置
            cfg = Config()
            cfg.epochs = args.epochs
            cfg.batch_size = args.batch_size
            cfg.hidden_dim = args.hidden_dim
            cfg.device = args.device
            # 设置消融开关
            cfg.use_rl = variant_config['use_rl']
            cfg.use_topology = variant_config['use_topology']
            cfg.use_multimodal = variant_config['use_multimodal']

            # 构造RL-TDAGCN模型（通过开关控制启用/禁用子模块）
            model = RLTDAGCN(cfg, adj_static).to(device)

            # 训练+评估
            metrics = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                adj_static, scaler_stats, cfg, device,
                variant_name, args.save_dir,
                has_weather=has_weather and cfg.use_multimodal,
                exp_id=exp_id, resume=resume
            )
            results[variant_name] = metrics

            # [v4] 每完成一个variant就保存
            save_partial_results(args.save_dir, exp_id, results)
            print(f"  [保存] {variant_name} 结果已保存")

    return results


def print_summary(exp_id, results):
    """
    打印实验结果汇总表
    """
    print(f"\n{'='*60}")
    print(f"  实验{exp_id} 结果汇总")
    print(f"{'='*60}")
    print(f"{'模型':<15} {'MAE':<10} {'RMSE':<10} {'SMAPE':<10}")
    print('-' * 45)
    for name, m in results.items():
        print(f"{name:<15} {m['MAE']:<10.4f} {m['RMSE']:<10.4f} {m['SMAPE']*100:<9.2f}%")


def main():
    parser = argparse.ArgumentParser(description='RL-TDAGCN 实验脚本 (v4: 断点续训+AMP)')
    parser.add_argument('--experiment', type=int, default=1,
                        help='实验编号(1-5), 或传入 --all 运行全部')
    parser.add_argument('--all', action='store_true',
                        help='运行全部5组实验')
    parser.add_argument('--dataset', type=str, default='./data/PEMS04',
                        help='数据目录路径')
    parser.add_argument('--epochs', type=int, default=150,
                        help='训练轮数 (v5: 100->150)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='隐藏层维度')
    parser.add_argument('--device', type=str, default='cpu',
                        help='训练设备 (cpu / cuda)')
    parser.add_argument('--save_dir', type=str, default='experiments/results',
                        help='结果保存目录')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式 (5个epoch，用于调试)')
    parser.add_argument('--resume', action='store_true',
                        help='[v4新增] 断点续训: 从上次中断处继续, 跳过已完成的variant')
    args = parser.parse_args()

    # 快速模式: 减少训练轮数和早停耐心值
    if args.quick:
        args.epochs = 5

    device = torch.device(args.device)

    # ==========================================
    # 加载数据
    # ==========================================
    print("=" * 60)
    print("  RL-TDAGCN 实验脚本 (v4: 断点续训 + AMP)")
    print("=" * 60)

    print("\n加载数据...")

    # 创建基础配置（用于数据加载）
    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.hidden_dim = args.hidden_dim
    config.device = args.device

    # 自动检测是否有天气数据
    weather_path = os.path.join(args.dataset, 'weather_aligned.npy')
    has_weather_data = os.path.exists(weather_path)
    if has_weather_data:
        print(f"  检测到天气数据: {weather_path}")
    config.use_multimodal = has_weather_data  # 数据加载时需要此标志

    # 创建数据加载器 (v4: 已在dataloader.py中添加num_workers和pin_memory)
    train_loader, val_loader, test_loader, adj_mx, scaler_stats = \
        create_data_loaders(args.dataset, config)
    adj_static = torch.FloatTensor(adj_mx).to(device)

    # ==========================================
    # 确定要运行的实验编号列表
    # ==========================================
    if args.all:
        exp_ids = list(EXPERIMENT_CONFIGS.keys())
    else:
        exp_ids = [args.experiment]

    # ==========================================
    # 逐个运行实验
    # ==========================================
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENT_CONFIGS:
            print(f"  [警告] 未知实验编号: {exp_id}, 跳过")
            continue

        exp_config = EXPERIMENT_CONFIGS[exp_id]
        print(f"\n{'='*60}")
        print(f"  实验{exp_id}: {exp_config['name']}")
        print(f"{'='*60}")

        # 运行实验 (v4: 传入resume参数)
        results = run_single_experiment(
            exp_id, exp_config,
            train_loader, val_loader, test_loader,
            adj_static, adj_mx, scaler_stats,
            device, args, resume=args.resume
        )

        # 保存最终结果
        os.makedirs(args.save_dir, exist_ok=True)
        result_file = os.path.join(args.save_dir, f'exp{exp_id}_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 打印汇总表
        print_summary(exp_id, results)
        print(f"\n[保存] 结果已保存至: {result_file}")

    print(f"\n{'='*60}")
    print("  所有实验完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
