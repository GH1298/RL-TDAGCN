"""
基础编码器模块
包含:
  1. DiffusionConv: 扩散卷积（前向+后向K阶扩散）
  2. SpatialEncoder: 空间编码器（多层扩散卷积 + 残差连接）
  3. TemporalEncoder: 时序编码器（GRU + LayerNorm）
  
扩散卷积是 DCRNN 的核心，能同时建模双向交通流传播:
  前向扩散: 模拟交通拥堵的顺流传播
  后向扩散: 模拟拥堵影响的回溯传播
  
数学表达:
  X_out = GRU(X_in) 
  H_spatial = DiffusionConv(H, A) = sum_{k=0}^{K-1} (P_forward^k + P_backward^k) * H * theta_k
  其中 P = D^{-1}A 是转移概率矩阵
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionConv(nn.Module):
    """
    扩散卷积层 (Diffusion Convolution)
    
    基于转移概率矩阵 P = D^{-1}A 的K阶截断扩散。
    分别计算前向和后向扩散，捕捉双向交通流传播。
    
    Args:
        in_channels: 输入特征维度
        out_channels: 输出特征维度
        K: 扩散阶数（截断步数）
    """
    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 前向和后向扩散各有一组权重
        # 每阶扩散一个权重矩阵: [K, in_channels, out_channels]
        self.weights_forward = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.weights_backward = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Kaiming 初始化"""
        for k in range(self.K):
            nn.init.kaiming_uniform_(self.weights_forward[k], a=1)
            nn.init.kaiming_uniform_(self.weights_backward[k], a=1)
        nn.init.zeros_(self.bias)
    
    def _compute_transition(self, adj):
        """
        计算转移概率矩阵 P = D^{-1}A
        
        Args:
            adj: [N, N] 邻接矩阵
        
        Returns:
            P: [N, N] 转移概率矩阵
            P_backward: [N, N] 逆转移概率矩阵
        """
        # 度矩阵
        deg = adj.sum(dim=1, keepdim=True) + 1e-8  # [N, 1]
        # 前向转移: P = D^{-1}A
        P = adj / deg  # [N, N]
        # 后向转移: P_backward = D^{-1}A^T
        P_backward = adj.t() / (adj.t().sum(dim=1, keepdim=True) + 1e-8)  # [N, N]
        return P, P_backward
    
    def forward(self, x, adj):
        """
        扩散卷积前向传播
        
        Args:
            x: [B, N, in_channels] 节点特征
            adj: [N, N] 邻接矩阵
        
        Returns:
            out: [B, N, out_channels] 扩散卷积输出
        """
        B, N, _ = x.shape
        P, P_backward = self._compute_transition(adj)  # [N, N]
        
        out = torch.zeros(B, N, self.out_channels, device=x.device)
        
        # K阶截断扩散
        P_fwd_power = torch.eye(N, device=x.device)  # P^0 = I
        P_bwd_power = torch.eye(N, device=x.device)  # P_backward^0 = I
        
        for k in range(self.K):
            # 前向扩散: P^k * X * theta_k
            x_fwd = torch.matmul(P_fwd_power, x)  # [B, N, in_channels]
            out += torch.matmul(x_fwd, self.weights_forward[k])
            
            # 后向扩散: P_backward^k * X * theta_k
            x_bwd = torch.matmul(P_bwd_power, x)
            out += torch.matmul(x_bwd, self.weights_backward[k])
            
            # 更新幂次
            if k < self.K - 1:
                P_fwd_power = torch.matmul(P_fwd_power, P)
                P_bwd_power = torch.matmul(P_bwd_power, P_backward)
        
        out += self.bias
        return out


class SpatialEncoder(nn.Module):
    """
    空间编码器
    多层扩散卷积 + 残差连接 + LayerNorm
    
    Args:
        input_dim: 输入特征维度（交通特征维度，如3）
        hidden_dim: 隐藏层维度
        num_layers: 扩散卷积层数
        K: 每层扩散阶数
        dropout: Dropout率
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, K=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 扩散卷积层
        self.diff_convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            self.diff_convs.append(
                DiffusionConv(hidden_dim, hidden_dim, K=K)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        Args:
            x: [B, T, N, F] 交通数据
            adj: [N, N] 邻接矩阵
        Returns:
            h: [B, T, N, hidden_dim] 空间编码特征
        """
        B, T, N, C = x.shape
        
        # 输入投影
        h = self.input_proj(x)  # [B, T, N, hidden_dim]
        h = F.relu(h)
        
        # 逐时间步应用扩散卷积
        h_list = []
        for t in range(T):
            h_t = h[:, t, :, :]  # [B, N, hidden_dim]
            for i in range(self.num_layers):
                # 残差连接
                residual = h_t
                h_t = self.diff_convs[i](h_t, adj)
                h_t = self.layer_norms[i](h_t + residual)
                h_t = self.dropout(h_t)
            h_list.append(h_t)

        # 使用 stack 避免 inplace 操作导致的计算图版本冲突
        h = torch.stack(h_list, dim=1)  # [B, T, N, hidden_dim]

        return h


class TemporalEncoder(nn.Module):
    """
    时序编码器
    使用GRU捕捉时序依赖，LayerNorm稳定训练
    
    Args:
        input_dim: 输入特征维度（hidden_dim）
        hidden_dim: GRU隐藏维度
        num_layers: GRU层数
        dropout: Dropout率
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GRU 时序建模
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, N, D] 空间编码后的特征
        
        Returns:
            h: [B, T, N, hidden_dim] 时序编码特征
        """
        B, T, N, D = x.shape
        
        # 将 N 维度展平到 batch 维度，方便GRU处理
        x_flat = x.reshape(B * N, T, D)  # [B*N, T, D]
        
        # GRU 编码
        h_flat, _ = self.gru(x_flat)  # [B*N, T, hidden_dim]
        
        # LayerNorm + Dropout
        h_flat = self.layer_norm(h_flat)
        h_flat = self.dropout(h_flat)
        
        # 恢复维度
        h = h_flat.reshape(B, T, N, self.hidden_dim)  # [B, T, N, hidden_dim]
        
        return h


class DAGCNEncoder(nn.Module):
    """
    DAGCN 完整编码器
    串联 SpatialEncoder + TemporalEncoder
    
    输入原始交通数据 -> 空间扩散卷积编码 -> GRU时序编码 -> 节点嵌入
    
    Args:
        input_dim: 交通特征维度
        hidden_dim: 隐藏层维度
        num_spatial_layers: 空间扩散卷积层数
        num_temporal_layers: GRU层数
        K: 扩散阶数
        dropout: Dropout率
    """
    def __init__(self, input_dim, hidden_dim=64, num_spatial_layers=2,
                 num_temporal_layers=2, K=2, dropout=0.3):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(
            input_dim, hidden_dim, num_spatial_layers, K, dropout
        )
        self.temporal_encoder = TemporalEncoder(
            hidden_dim, hidden_dim, num_temporal_layers, dropout
        )
    
    def forward(self, traffic, adj):
        """
        Args:
            traffic: [B, T, N, F] 原始交通数据
            adj: [N, N] 邻接矩阵（可以是静态图或RL优化后的动态图）
        
        Returns:
            h: [B, T, N, hidden_dim] 编码后的节点特征
        """
        # 空间编码
        h = self.spatial_encoder(traffic, adj)  # [B, T, N, H]
        # 时序编码
        h = self.temporal_encoder(h)  # [B, T, N, H]
        return h