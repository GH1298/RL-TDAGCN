"""
拓扑约束模块 (Topology Constraint)
包含:
  1. TopologyEncoder: 从静态邻接矩阵提取拓扑结构先验
  2. TopologyConstraint: 约束动态图不偏离交通道路结构

设计思路:
  - 动态图 A_t 是完全数据驱动的，可能学习到"预测正确但交通不合理"的关系
  - 拓扑约束通过两种方式限制:
    (1) Topology Loss: L_topo = ||A_t - A_static||_F^2，惩罚偏离
    (2) Topology Attention: A = beta * A_t + (1-beta) * A'_topo，结构融合

数学表达:
  T = GCN(A_static)                          # 拓扑嵌入
  A'_topo = softmax(T * T^T)                 # 拓扑引导的邻接矩阵
  L_topo = ||A_t - A'_topo||_F^2             # 拓扑损失
  A_final = beta * A_t + (1-beta) * A'_topo  # 拓扑注意力融合

v2 改进日志 (2026-07-31):
  - Beta初始化: 改用 inverse-sigmoid(logit) 精确映射到 fusion_beta，并 clamp 防止极端值
  - 拓扑损失: 由普通MSE改为加权MSE，真实道路连接(非零)权重更高
  - 道路等级拓扑: 新增 set_road_type_adj，支持融合高德道路等级邻接矩阵
  - 可视化诊断: 新增 visualize_adjacency 热力图对比方法
  - Beta监控: 新增 get_current_beta，forward 存储当前beta供训练循环读取
    (forward 仍返回2值tuple，完全向后兼容 rlt_dagcn.py)
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyEncoder(nn.Module):
    """
    拓扑编码器 (v3增强)
    
    使用多层GCN在静态邻接矩阵上传播，提取道路网络的结构特征。
    
    v3改进:
      1. 支持可变GCN层数(2->3)，更深层的拓扑特征提取
      2. GCN层间加入残差连接，防止深层梯度消失
      3. 加入度中心性先验: 将节点度作为结构特征注入嵌入
    
    输出的拓扑嵌入用于:
    1. 生成拓扑引导的邻接矩阵 A'_topo
    2. 作为RL状态的一部分
    
    Args:
        num_nodes: 节点数量
        embed_dim: 拓扑嵌入维度
        num_layers: GCN层数 (v3默认3层)
    """
    def __init__(self, num_nodes, embed_dim=16, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 节点标识嵌入（可学习）
        self.node_id_embed = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
        
        # [v3新增] 度中心性嵌入: 将节点度(连接数)作为结构先验
        # 理论: 交通路网中，高度节点(枢纽/交叉路口)和低度节点(末端道路)
        #       具有不同的拓扑角色，应该有不同的嵌入特征
        self.degree_embed = nn.Linear(1, embed_dim)
        
        # GCN层提取拓扑特征
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else embed_dim
            self.gcn_layers.append(nn.Linear(in_dim, embed_dim))
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, static_adj):
        """
        Args:
            static_adj: [N, N] 静态邻接矩阵
        
        Returns:
            topo_embed: [N, embed_dim] 拓扑嵌入
            topo_adj: [N, N] 拓扑引导的邻接矩阵
        """
        N = static_adj.shape[0]
        
        # === [v3新增] 计算度中心性并注入嵌入 ===
        # 度 = 每行非零元素个数（道路连接数）
        degree = (static_adj > 0).float().sum(dim=1, keepdim=True)  # [N, 1]
        degree_norm = degree / (degree.max() + 1e-8)  # 归一化到 [0, 1]
        degree_feat = self.degree_embed(degree_norm)  # [N, embed_dim]
        
        # 融合节点ID嵌入和度中心性嵌入
        h = self.node_id_embed + degree_feat  # [N, embed_dim]
        
        # GCN 传播: H^(l+1) = A * H^(l) * W
        # [v3新增] 残差连接: 深层GCN中防止梯度消失
        for i, gcn in enumerate(self.gcn_layers):
            # 图卷积: 邻居信息聚合
            h_new = torch.matmul(static_adj, h)  # [N, embed_dim]
            h_new = gcn(h_new)
            h_new = F.relu(h_new)
            # 残差连接(从第2层开始): h_new = h_new + h
            if i > 0:
                h_new = h_new + h
            h = h_new
        
        h = self.layer_norm(h)
        topo_embed = h  # [N, embed_dim]
        
        # 从拓扑嵌入生成拓扑引导邻接矩阵
        topo_adj = torch.mm(topo_embed, topo_embed.t())  # [N, N]
        topo_adj = F.softmax(topo_adj, dim=-1)  # 行归一化
        
        return topo_embed, topo_adj


class TopologyConstraint(nn.Module):
    """
    拓扑约束模块 (改进版)
    
    组合 Topology Loss 和 Topology Attention 两种约束方式。
    
    改进点:
      1. Beta初始化: 用 logit(fusion_beta) 作为 beta_raw 初值，使
         sigmoid(beta_raw) 恰好等于 fusion_beta，并 clamp 在 [0.1, 0.95]
         防止训练漂移到极端值。
      2. 拓扑损失: 改为加权MSE，对真实道路连接(static_adj非零)权重更高。
      3. 道路等级拓扑: set_road_type_adj 注入高德道路等级邻接矩阵，
         与GCN拓扑各占50%融合 (不设置时为旧版纯GCN拓扑，向后兼容)。
      4. 可视化: visualize_adjacency 输出动态图/拓扑图/融合图热力图。
      5. Beta监控: get_current_beta 读取最近一次forward的beta值，
         forward 返回值仍为2值tuple，完全兼容现有调用方。
    
    Args:
        num_nodes: 节点数量
        embed_dim: 拓扑嵌入维度
        num_gcn_layers: 拓扑GCN层数
        loss_weight: 拓扑损失权重 lambda_topo
        fusion_beta: 拓扑融合参数初始值（可学习），beta * A_dyn + (1-beta) * A_topo
        learnable_beta: 是否让beta可学习
    """
    def __init__(self, num_nodes, embed_dim=16, num_gcn_layers=2,
                 loss_weight=0.1, fusion_beta=0.7, learnable_beta=True):
        super().__init__()
        self.loss_weight = loss_weight
        
        # 拓扑编码器
        self.topo_encoder = TopologyEncoder(num_nodes, embed_dim, num_gcn_layers)
        
        # 可学习的融合参数 beta
        # 改进: 使用 inverse sigmoid (logit) 把 fusion_beta 精确映射到 beta_raw 初值
        if learnable_beta:
            self._init_beta = fusion_beta
            # logit(p) = log(p/(1-p))，其逆运算是 sigmoid
            # sigmoid(logit(fusion_beta)) == fusion_beta，训练初始即为目标融合比例
            # 例: fusion_beta=0.9 -> logit(0.9)=log(0.9/0.1)≈2.197
            if 0.0 < fusion_beta < 1.0:
                beta_init_raw = math.log(fusion_beta / (1.0 - fusion_beta))
            else:
                # 边界值(0或1)无法用logit表示，回退到0(对应sigmoid(0)=0.5)
                beta_init_raw = 0.0
            self.beta_raw = nn.Parameter(torch.tensor(beta_init_raw))
        else:
            # 不可学习模式: 直接使用固定beta
            self.register_buffer('beta', torch.tensor(fusion_beta))
            self._init_beta = fusion_beta
            self.beta_raw = None
        
        # 道路等级拓扑邻接矩阵 (可选, 来自高德数据)
        # 默认None表示不启用，行为与旧版一致(纯GCN拓扑)
        self._road_type_adj = None
        
        # 当前beta值缓存 (供训练循环监控, 不修改forward返回值个数)
        self._current_beta = None
        
        # 缓存拓扑特征（静态图不变，只需计算一次）
        # 注意: 缓存时必须 detach，否则会保留旧计算图导致 backward 二次访问报错
        self._topo_cached = None
        self._static_adj_cached = None
    
    def _get_beta(self):
        """
        获取当前融合参数 beta
        
        改进说明:
          - 直接用 sigmoid(beta_raw) 映射到 (0,1)，不再乘 2*_init_beta
            (旧版 sigmoid(0)*2*0.9=0.9 看似正确，但 beta_raw 一旦学到负值
             会使 beta 迅速变小，约束力下降)
          - clamp 在 [0.1, 0.95] 防止训练漂移到极端值:
            下限0.1: 保证拓扑图至少有10%影响力，避免拓扑约束形同虚设
            上限0.95: 保证动态图至少有5%影响力，保留数据驱动能力
        
        Returns:
            beta: scalar tensor, 范围 [0.1, 0.95]
        """
        if self.beta_raw is not None:
            # sigmoid 映射到 (0,1)，再 clamp 限制在 [0.1, 0.95]
            return torch.clamp(torch.sigmoid(self.beta_raw), 0.1, 0.95)
        return self.beta
    
    def set_road_type_adj(self, road_type_adj):
        """
        设置道路等级邻接矩阵 (来自高德数据)
        
        设置后，拓扑约束的引导图由两部分融合而成:
            topo_adj = 0.5 * gcn_distance_topo + 0.5 * road_type_topo
        - gcn_distance_topo: TopologyEncoder 基于静态(距离/连接)图生成的拓扑
        - road_type_topo:    高德道路等级相似度构建的邻接矩阵
        
        未调用本方法时，仅使用 GCN 距离拓扑 (向后兼容旧版行为)。
        
        Args:
            road_type_adj: [N, N] 道路等级邻接矩阵
                           (numpy.ndarray 或 torch.Tensor 均可)
        """
        # 统一转为 torch.Tensor
        if isinstance(road_type_adj, np.ndarray):
            road_type_adj = torch.from_numpy(road_type_adj).float()
        else:
            road_type_adj = road_type_adj.float()
        
        # 注册为 buffer: 随模型保存/加载，且会随 .to(device) 自动迁移设备
        # 重复设置时先移除旧 buffer 避免 warning
        if 'road_type_adj' in self._buffers:
            del self._buffers['road_type_adj']
        self.register_buffer('road_type_adj', road_type_adj)
        # 标记已启用道路等级拓扑
        self._road_type_adj = self.road_type_adj
    
    def forward(self, dynamic_adj, static_adj, cache=True):
        """
        应用拓扑约束
        
        Args:
            dynamic_adj: [B, N, N] 或 [N, N] 动态邻接矩阵
            static_adj: [N, N] 静态邻接矩阵
            cache: 是否缓存拓扑特征（保留参数以兼容调用方，当前未启用缓存）
        
        Returns:
            constrained_adj: [B, N, N] 或 [N, N] 拓扑约束后的邻接矩阵
            topo_loss: scalar 已乘以 loss_weight 的拓扑约束损失
        """
        # 计算拓扑特征 (GCN在静态图上传播，生成距离/结构拓扑引导图)
        # 注: 拓扑编码器含可学习参数，每次forward都需参与梯度计算，故不缓存
        topo_embed, topo_adj = self.topo_encoder(static_adj)
        
        # === 道路等级拓扑融合 (可选) ===
        # 若注入了高德道路等级邻接矩阵，则与GCN拓扑各占50%融合
        if self._road_type_adj is not None:
            road_adj = self._road_type_adj.to(topo_adj.device)
            # 对齐形状后再融合
            topo_adj = 0.5 * topo_adj + 0.5 * road_adj
        
        # === 拓扑损失 (加权MSE) ===
        # 改进: 对 static_adj 中非零元素(真实道路连接)给予更高权重
        # 有道路连接的节点对权重=3，无连接权重=1
        # 这样拓扑约束更关注"真实道路连接关系"的保持，而非均匀惩罚所有节点对
        weight_mask = (static_adj > 0).float() * 2.0 + 1.0  # [N, N]
        
        if dynamic_adj.dim() == 3:
            # [B, N, N] -> 对每个batch计算加权loss
            topo_target = topo_adj.unsqueeze(0).expand_as(dynamic_adj)
            weight_mask_expanded = weight_mask.unsqueeze(0).expand_as(dynamic_adj)
            topo_loss = ((dynamic_adj - topo_target) ** 2 * weight_mask_expanded).mean()
        else:
            # [N, N]
            topo_loss = ((dynamic_adj - topo_adj) ** 2 * weight_mask).mean()
        
        # === 拓扑注意力融合 ===
        beta = self._get_beta()
        # 缓存当前beta值，供训练循环通过 get_current_beta() 监控
        # (不修改forward返回值个数，保持与 rlt_dagcn.py 的2值解包兼容)
        self._current_beta = beta.item()
        
        if dynamic_adj.dim() == 3:
            constrained_adj = beta * dynamic_adj + (1 - beta) * topo_adj.unsqueeze(0)
        else:
            constrained_adj = beta * dynamic_adj + (1 - beta) * topo_adj
        
        return constrained_adj, topo_loss * self.loss_weight
    
    def get_current_beta(self):
        """
        获取当前融合beta值 (供训练循环调用)
        
        每次 forward 调用后会更新 self._current_beta。
        若 forward 尚未执行过，则返回 None。
        
        Returns:
            float: 当前beta值 (范围 [0.1, 0.95])，或 None
        """
        return getattr(self, '_current_beta', None)
    
    def visualize_adjacency(self, dynamic_adj, static_adj, epoch, save_dir):
        """
        生成邻接矩阵热力图对比，用于诊断拓扑约束效果
        
        并排绘制3张热力图:
          1. dynamic_adj - 动态图 (RL优化后、拓扑约束前)
          2. topo_adj     - 拓扑引导图 (GCN + 可选道路等级)
          3. A_final      - 拓扑融合后的最终图
        
        图片保存到 {save_dir}/topo_epoch_{epoch}.png
        
        Args:
            dynamic_adj: [B, N, N] 或 [N, N] 动态邻接矩阵
            static_adj:  [N, N] 静态邻接矩阵
            epoch:       当前训练轮次 (用于文件命名)
            save_dir:    图片保存目录
        
        Returns:
            save_path: str 实际保存路径
        """
        # 延迟导入可视化工具，避免在未安装matplotlib时影响模块加载
        from utils.topo_visualizer import plot_topo_comparison
        
        with torch.no_grad():
            # 复算拓扑引导图 (与forward一致: GCN拓扑 + 可选道路等级融合)
            _, topo_adj = self.topo_encoder(static_adj)
            if self._road_type_adj is not None:
                road_adj = self._road_type_adj.to(topo_adj.device)
                topo_adj = 0.5 * topo_adj + 0.5 * road_adj
            
            # 当前融合beta
            beta = self._get_beta()
            
            # 取单个样本用于可视化 ([B,N,N] -> [N,N])
            if dynamic_adj.dim() == 3:
                dyn = dynamic_adj[0]
            else:
                dyn = dynamic_adj
            # 最终融合图 (与forward公式一致)
            final_adj = beta * dyn + (1 - beta) * topo_adj
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 调用可视化工具绘制三图对比并保存
        plot_topo_comparison(
            dynamic_adj=dyn.detach().cpu().numpy(),
            topo_adj=topo_adj.detach().cpu().numpy(),
            final_adj=final_adj.detach().cpu().numpy(),
            epoch=epoch,
            save_dir=save_dir,
        )
        save_path = os.path.join(save_dir, f'topo_epoch_{epoch}.png')
        return save_path
    
    def get_topo_embed(self, static_adj):
        """获取拓扑嵌入（用于RL状态）"""
        topo_embed, _ = self.topo_encoder(static_adj)
        return topo_embed
