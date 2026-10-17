"""
RL-TDAGCN 完整模型组装文件

这是整个论文的核心模型文件，将所有子模块按论文架构图正确串联。

整体架构:
  1. 第一阶段编码: 使用当前图(prev_adj)对交通数据进行DAGCN编码
  2. 动态图生成: 基于编码特征生成候选动态图
  3. RL图控制: PPO控制器决定图的更新策略
  4. 拓扑约束: 约束动态图不偏离道路结构
  5. 第二阶段编码: 使用优化后的图进行空间精炼
  6. 多模态融合: (可选) 融合天气信息
  7. 预测输出: 从精炼特征生成预测结果

设计决策:
  - RL 控制的是整个 batch 共享的图 [N, N]，而非每个样本独立图
  - 状态在 batch 内聚合为单个向量，RL 输出单个动作
  - 两阶段编码: 先用当前图编码 -> 生成动态图 -> RL优化 -> 拓扑约束 -> 用优化后的图做最终空间精炼
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_encoder import DAGCNEncoder, SpatialEncoder
from models.dynamic_graph import DynamicGraphGenerator
from models.topology_constraint import TopologyConstraint
from models.rl_graph_controller import PPOGraphController, RolloutBuffer
from models.multimodal_fusion import WeatherEncoder, GatingFusion


class RLTDAGCN(nn.Module):
    """
    RL-TDAGCN 完整模型

    将动态图生成、RL图控制、拓扑约束、多模态融合等子模块
    整合为统一的交通预测模型。

    模块开关（消融实验用）:
      - use_rl:          是否启用RL图控制器（False时直接用A_candidate）
      - use_topology:    是否启用拓扑约束
      - use_multimodal:  是否启用多模态融合
      - use_dynamic_graph: 是否使用动态图（False时只用静态图）

    Args:
        config: 配置对象，包含所有超参数
        adj_static: [N, N] 静态邻接矩阵
    """

    def __init__(self, config, adj_static):
        super().__init__()

        # ========== 基本配置 ==========
        self.config = config
        self.num_nodes = config.num_nodes
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        # ========== 模块开关（消融实验用） ==========
        self.use_rl = getattr(config, 'use_rl', True)
        self.use_topology = getattr(config, 'use_topology', True)
        self.use_multimodal = getattr(config, 'use_multimodal', False)
        self.use_dynamic_graph = getattr(config, 'use_dynamic_graph', True)

        # ========== 第一阶段: DAGCN编码器 ==========
        # 包含 SpatialEncoder(扩散卷积) + TemporalEncoder(GRU时序建模)
        self.encoder = DAGCNEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_spatial_layers=getattr(config, 'num_layers', 2),
            num_temporal_layers=getattr(config, 'num_layers', 2),
            K=getattr(config, 'K', 2),
            dropout=getattr(config, 'dropout', 0.3),
        )

        # ========== 动态图生成器 ==========
        # 基于多头注意力机制，从节点特征生成候选动态邻接矩阵
        self.graph_generator = DynamicGraphGenerator(
            input_dim=self.hidden_dim,
            embed_dim=getattr(config, 'graph_embed_dim', 10),
            num_heads=getattr(config, 'graph_heads', 4),
            temperature=getattr(config, 'graph_temp', 0.5),
        )

        # ========== 第二阶段: 空间精炼编码器 ==========
        # 使用优化后的图(A_final)重新进行空间编码，精炼节点特征
        # 输入已是 hidden_dim 空间，故 input_dim 设为 hidden_dim
        self.spatial_refiner = SpatialEncoder(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=getattr(config, 'num_layers', 2),
            K=getattr(config, 'K', 2),
            dropout=getattr(config, 'dropout', 0.3),
        )

        # ========== RL 图控制器 ==========
        if self.use_rl:
            # PPO控制器: 状态维度固定为6维
            self.rl_controller = PPOGraphController(
                state_dim=6,
                hidden_dim=getattr(config, 'rl_hidden_dim', 64),
                num_freq_actions=3,  # 0=不更新, 1=普通更新, 2=快速更新
                clip_epsilon=getattr(config, 'rl_clip_epsilon', 0.2),
                entropy_coef=getattr(config, 'rl_entropy_coef', 0.01),
                value_coef=getattr(config, 'rl_value_coef', 0.5),
                gae_lambda=getattr(config, 'rl_gae_lambda', 0.95),
                gamma=getattr(config, 'rl_gamma', 0.99),
            )
            # 经验回放缓冲区
            self.rl_buffer = RolloutBuffer()

        # ========== 拓扑约束模块 ==========
        if self.use_topology:
            self.topo_constraint = TopologyConstraint(
                num_nodes=self.num_nodes,
                embed_dim=getattr(config, 'graph_embed_dim', 10),
                num_gcn_layers=getattr(config, 'topo_gcn_layers', 2),
                loss_weight=getattr(config, 'topo_loss_weight', 0.1),
                fusion_beta=getattr(config, 'topo_fusion_beta', 0.7),
            )

        # ========== 多模态融合模块 ==========
        if self.use_multimodal:
            self.weather_encoder = WeatherEncoder(
                weather_dim=getattr(config, 'weather_dim', 6),
                hidden_dim=self.hidden_dim,
            )
            self.gating_fusion = GatingFusion(
                hidden_dim=self.hidden_dim,
                gate_hidden=getattr(config, 'fusion_gate_hidden', 32),
            )

        # ========== 输出预测层 ==========
        self.output_layer = nn.Linear(self.hidden_dim, self.pred_len)

        # ========== 注册缓冲区（不参与梯度计算但会随模型保存/加载） ==========
        # prev_adj: 上一步使用的邻接矩阵，初始为静态图
        self.register_buffer('prev_adj', adj_static.clone())
        # 保存原始静态图副本，用于 reset_graph_state 重置
        self.register_buffer('adj_static_orig', adj_static.clone())

        # RL 相关缓冲区
        if self.use_rl:
            # EMA预测误差: 用于构造RL状态中的 pred_error 维度
            self.register_buffer('ema_pred_error', torch.tensor(0.0))

    # ================================================================
    # 内部方法: RL状态构造
    # ================================================================
    def _construct_rl_state(self, traffic, A_candidate):
        """
        构造RL状态向量（6维）

        状态构成:
          1. speed_mean   - 速度均值（全batch聚合）
          2. speed_std     - 速度标准差
          3. flow_mean     - 流量均值
          4. pred_error    - 上一步预测误差（EMA跟踪）
          5. graph_change  - 图变化量 ||A_candidate - A_prev||_F
          6. graph_density - 图稀疏度（非对角线边权重均值）

        Args:
            traffic:     [B, T, N, F] 交通数据
            A_candidate: [N, N] 候选动态图

        Returns:
            state: [6] RL状态向量（标量值，已移至对应device）
        """
        # --- 交通统计 ---
        # 假设 traffic[..., 0] 是速度特征，traffic[..., 1] 是流量特征
        speed = traffic[:, :, :, 0]  # [B, T, N]
        speed_mean = speed.mean().item()
        speed_std = speed.std().clamp(min=1e-6).item()

        if traffic.size(-1) >= 2:
            flow = traffic[:, :, :, 1]  # [B, T, N]
            flow_mean = flow.mean().item()
        else:
            flow_mean = 0.0

        # --- 图统计 ---
        # 图变化量: 候选图与上一步图的Frobenius范数差异
        graph_change = torch.norm(A_candidate - self.prev_adj, p='fro').item()

        # 图稀疏度: 非对角线元素均值（衡量图的整体密度）
        N = A_candidate.size(0)
        mask = ~torch.eye(N, dtype=torch.bool, device=A_candidate.device)
        graph_density = A_candidate[mask].mean().item()

        # --- 预测误差（EMA跟踪） ---
        pred_error = self.ema_pred_error.item()

        # 组装状态向量（所有值已通过.item()转为Python标量，确保shape一致）
        state = torch.tensor([
            speed_mean, speed_std, flow_mean,
            pred_error, graph_change, graph_density
        ], dtype=torch.float32, device=traffic.device)

        return state

    # ================================================================
    # 内部方法: RL动作应用
    # ================================================================
    def _apply_rl_action(self, A_candidate, action_disc, action_cont):
        """
        根据RL动作更新图

        动作空间:
          - action_disc (离散): 更新频率
            0 = 不更新（完全保持上一步的图）
            1 = 普通更新（按alpha加权融合新旧图）
            2 = 快速更新（直接使用候选图替换）
          - action_cont (连续): 更新比例 alpha in [0, 1]

        Args:
            A_candidate: [N, N] 候选动态图
            action_disc: scalar 离散动作
            action_cont: scalar 连续动作（alpha）

        Returns:
            A_optimized: [N, N] RL优化后的图
        """
        freq = action_disc.item() if isinstance(action_disc, torch.Tensor) else int(action_disc)
        alpha = action_cont.item() if isinstance(action_cont, torch.Tensor) else float(action_cont)
        # 确保 alpha 在 [0, 1] 范围内
        alpha = max(0.0, min(1.0, alpha))

        if freq == 0:
            # 不更新: 完全保持上一步的图
            A_optimized = self.prev_adj.clone()
        elif freq == 1:
            # 普通更新: 按alpha比例加权融合新旧图
            A_optimized = (1.0 - alpha) * self.prev_adj + alpha * A_candidate
        else:  # freq == 2
            # 快速更新: 直接使用候选图
            A_optimized = A_candidate.clone()

        # 确保对角线为1（自环连接，保证节点自身信息传递）
        diag_idx = torch.arange(A_optimized.size(0), device=A_optimized.device)
        A_optimized[diag_idx, diag_idx] = 1.0

        return A_optimized

    # ================================================================
    # 前向传播
    # ================================================================
    def forward(self, traffic, weather, adj_static, target=None, collect_rl=False):
        """
        完整前向传播

        流程:
          Step 1: DAGCNEncoder(traffic, current_adj) -> h [B, T, N, H]
          Step 2: 取最后时间步 -> batch均值 -> DynamicGraphGenerator -> A_candidate [N, N]
          Step 3: (if use_rl) 构造RL状态 -> PPO选择动作 -> 更新图
          Step 4: (if use_topology) 拓扑约束 -> A_final
          Step 5: (else) A_final = A_candidate 或 adj_static
          Step 6: (if use_multimodal) 天气编码 -> 门控融合
          Step 7: 第二阶段空间精炼 -> 最终预测
          Step 8: 计算损失

        Args:
            traffic:    [B, T, N, F] 交通数据
            weather:    [B, T, Fw] 天气数据（可选，None表示不使用）
            adj_static: [N, N] 静态邻接矩阵
            target:     [B, pred_len, N] 真实标签（用于计算损失，可选）
            collect_rl:  bool 是否收集RL经验到buffer（训练时为True）

        Returns:
            pred:      [B, pred_len, N] 预测结果
            info_dict: dict 包含各种损失和调试信息
        """
        B, T, N, Fea = traffic.shape  # Fea: 特征维度（避免遮蔽 F=nn.functional）
        device = traffic.device

        # 初始化信息字典（用于调试和训练监控）
        info = {
            'pred_loss': torch.tensor(0.0, device=device),
            'topo_loss': torch.tensor(0.0, device=device),
            'total_loss': torch.tensor(0.0, device=device),
            'rl_data': None,
            'action_disc': None,
            'action_cont': None,
            'action_desc': '',
            'A_candidate': None,
            'A_final': None,
        }

        # ==========================================
        # Step 1: 第一阶段编码（使用当前图 prev_adj）
        # ==========================================
        current_adj = self.prev_adj  # [N, N], 上一步优化后的图
        h = self.encoder(traffic, current_adj)  # [B, T, N, H]

        # ==========================================
        # Step 2: 动态图生成
        # ==========================================
        if self.use_dynamic_graph:
            # 取最后时间步的节点特征
            h_last = h[:, -1, :, :]  # [B, N, H]

            # Batch维度聚合: 对所有样本取均值
            # 核心设计: RL控制的是整个batch共享的图，不是每个样本独立图
            h_agg = h_last.mean(dim=0, keepdim=True)  # [1, N, H]

            # 生成候选动态图
            A_candidate_batch = self.graph_generator(h_agg)  # [1, N, N]
            A_candidate = A_candidate_batch.squeeze(0)  # [N, N]
        else:
            # 不使用动态图，直接使用静态图
            A_candidate = adj_static.clone()

        info['A_candidate'] = A_candidate.detach()

        # ==========================================
        # Step 3: RL 图控制（可选）
        # ==========================================
        if self.use_rl:
            # 构造6维RL状态
            state = self._construct_rl_state(traffic, A_candidate)  # [6]
            state = state.unsqueeze(0)  # [1, 6]（batch=1，图是batch共享的）

            # 训练时随机采样探索，推理时确定性选择
            deterministic = not collect_rl
            action_disc, action_cont, log_prob, value, entropy = \
                self.rl_controller.select_action(state, deterministic=deterministic)

            # 记录动作信息（用于日志和调试）
            info['action_disc'] = action_disc.detach()
            info['action_cont'] = action_cont.detach()
            info['action_desc'] = self.get_rl_action_description(
                action_disc.squeeze(0), action_cont.squeeze(0)
            )

            # 根据RL动作更新图
            A_optimized = self._apply_rl_action(
                A_candidate, action_disc.squeeze(0), action_cont.squeeze(0)
            )

            # 收集RL经验数据（供后续PPO更新使用）
            info['rl_data'] = {
                'state': state.detach(),
                'action_disc': action_disc.detach(),
                'action_cont': action_cont.detach(),
                'log_prob': log_prob.detach() if log_prob is not None else None,
                'value': value.detach() if value is not None else None,
                'entropy': entropy.detach() if entropy is not None else None,
            }
        else:
            # 不使用RL，直接使用候选图
            A_optimized = A_candidate

        # ==========================================
        # Step 4: 拓扑约束（可选）
        # ==========================================
        topo_loss = torch.tensor(0.0, device=device)

        if self.use_topology:
            # 传入 [N, N]，拓扑约束模块内部处理维度
            A_final, topo_loss = self.topo_constraint(A_optimized, adj_static, cache=True)
            info['topo_loss'] = topo_loss
        else:
            # 不使用拓扑约束，直接使用RL输出或候选图
            A_final = A_optimized

        info['A_final'] = A_final.detach()

        # ==========================================
        # Step 5: 确定最终使用的图
        # （已在 Step 3/4 中完成 A_final 的确定）
        # ==========================================

        # ==========================================
        # Step 6: 多模态融合（可选）
        # ==========================================
        if self.use_multimodal and weather is not None:
            # 天气特征编码
            w_feat = self.weather_encoder(weather)  # [B, T, H]

            # 门控融合: 自适应融合交通特征和天气特征
            h_for_refine = self.gating_fusion(h, w_feat)  # [B, T, N, H]
        else:
            # 不使用多模态，直接使用编码后的交通特征
            h_for_refine = h

        # ==========================================
        # Step 7: 第二阶段空间精炼 + 最终预测
        # ==========================================
        # 使用优化后的图 A_final 重新进行空间编码，精炼节点特征
        h_final = self.spatial_refiner(h_for_refine, A_final)  # [B, T, N, H]

        # 取最后时间步的特征用于预测
        h_pred = h_final[:, -1, :, :]  # [B, N, H]

        # 线性预测层: [B, N, H] -> [B, N, pred_len]
        pred = self.output_layer(h_pred)  # [B, N, pred_len]

        # 转置为 [B, pred_len, N]
        pred = pred.permute(0, 2, 1)  # [B, pred_len, N]

        # ==========================================
        # Step 8: 计算损失
        # ==========================================
        if target is not None:
            # 预测损失: MAE (Mean Absolute Error)
            pred_loss = F.l1_loss(pred, target)
            info['pred_loss'] = pred_loss

            # 总损失 = 预测损失 + 拓扑损失（RL损失由ppo_update单独计算）
            total_loss = pred_loss + topo_loss
            info['total_loss'] = total_loss

            # 更新EMA预测误差（用于下一步RL状态构造）
            if self.use_rl:
                with torch.no_grad():
                    step_error = pred_loss.item()
                    # 指数移动平均，平滑因子0.1
                    self.ema_pred_error = (0.9 * self.ema_pred_error
                                            + 0.1 * step_error)
        else:
            info['total_loss'] = torch.tensor(0.0, device=device)

        # 更新 prev_adj 为当前优化后的图（下一步forward使用）
        with torch.no_grad():
            self.prev_adj.copy_(A_final.detach())

        return pred, info

    # ================================================================
    # 推理方法
    # ================================================================
    def predict(self, traffic, weather, adj_static):
        """
        推理模式（确定性策略，不收集RL经验）

        用于测试和评估阶段，RL控制器使用确定性动作选择。
        内部调用 forward 但关闭梯度计算和RL探索。

        Args:
            traffic:    [1, T, N, F] 单样本交通数据
            weather:    [1, T, Fw] 天气数据（可选，可传None）
            adj_static: [N, N] 静态邻接矩阵

        Returns:
            pred: [1, pred_len, N] 预测结果
        """
        with torch.no_grad():
            pred, info = self.forward(
                traffic, weather, adj_static,
                target=None, collect_rl=False
            )
        return pred

    # ================================================================
    # 工具方法
    # ================================================================
    @staticmethod
    def get_rl_action_description(action_disc, action_cont):
        """
        将RL动作转为可读描述（用于日志记录）

        Args:
            action_disc: 离散动作 (0=不更新, 1=普通更新, 2=快速更新)
            action_cont: 连续动作 alpha (融合比例)

        Returns:
            desc: str 动作的可读描述
        """
        freq = action_disc.item() if isinstance(action_disc, torch.Tensor) else int(action_disc)
        alpha = action_cont.item() if isinstance(action_cont, torch.Tensor) else float(action_cont)

        freq_map = {
            0: '不更新(保持旧图)',
            1: f'普通更新(alpha={alpha:.3f})',
            2: '快速更新(替换新图)',
        }

        return freq_map.get(freq, f'未知动作(freq={freq})')

    def reset_graph_state(self):
        """
        重置图状态为静态邻接矩阵

        用途:
          - 新epoch开始时重置图状态
          - 评估/测试前恢复到初始状态
          - 图状态异常时恢复默认
        """
        # 将 prev_adj 重置为原始静态图
        self.prev_adj.copy_(self.adj_static_orig)

        # 重置 EMA 预测误差
        if self.use_rl:
            self.ema_pred_error.zero_()
