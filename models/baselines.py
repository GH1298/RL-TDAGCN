"""
基线模型集合
用于实验1的模型对比实验，所有基线模型统一接口:
  forward(traffic, adj_static) -> (pred, {})
  pred shape: [B, pred_len, N]

包含5个基线:
  1. DCRNNBaseline      - 固定图扩散卷积循环网络
  2. GraphWaveNet       - 自适应图扩散卷积网络
  3. STGCN              - 时空图卷积网络
  4. TGCN               - 时序图卷积网络 (GRU + GCN融合)
  5. LSTMModel          - 纯时序基线 (无图结构)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_encoder import DAGCNEncoder, DiffusionConv
from models.dynamic_graph import DynamicGraphGenerator


# ======================================================================
# 1. DCRNN (固定图基线)
# 直接复用 DAGCNEncoder，不使用动态图，只用静态图
# ======================================================================
class DCRNNBaseline(nn.Module):
    """
    DCRNN 基线模型
    使用 DAGCNEncoder（扩散卷积 + GRU）对交通数据进行编码，
    取最后时间步特征后通过线性层输出预测。

    与完整 RL-TDAGCN 的区别: 不使用动态图、RL控制器、拓扑约束、多模态融合。

    Args:
        input_dim:  输入特征维度 (如3: flow, occupancy, speed)
        hidden_dim: 隐藏层维度
        pred_len:   预测步数
    """

    def __init__(self, input_dim, hidden_dim, pred_len):
        super().__init__()
        # 复用 DAGCNEncoder: SpatialEncoder(DiffusionConv) + TemporalEncoder(GRU)
        self.encoder = DAGCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_spatial_layers=2,
            num_temporal_layers=2,
            K=2,
            dropout=0.3,
        )
        # 输出投影: [B, N, H] -> [B, N, pred_len] -> permute -> [B, pred_len, N]
        self.output_layer = nn.Linear(hidden_dim, pred_len)

    def forward(self, traffic, adj_static):
        """
        Args:
            traffic:    [B, T, N, F] 交通数据
            adj_static: [N, N]       静态邻接矩阵
        Returns:
            pred: [B, pred_len, N] 预测结果
            info: {} (空字典，统一接口)
        """
        # 编码: [B, T, N, F] -> [B, T, N, H]
        h = self.encoder(traffic, adj_static)
        # 取最后时间步: [B, N, H]
        h_last = h[:, -1, :, :]
        # 线性预测: [B, N, pred_len]
        pred = self.output_layer(h_last)
        # 转置为 [B, pred_len, N]
        pred = pred.permute(0, 2, 1)
        return pred, {}


# ======================================================================
# 2. GraphWaveNet (自适应图基线)
# 使用 DynamicGraphGenerator 学习自适应图，但不用RL控制
# ======================================================================
class GraphWaveNet(nn.Module):
    """
    GraphWaveNet 基线模型
    通过 DynamicGraphGenerator 学习自适应邻接矩阵，
    然后用自适应图做扩散卷积编码，GRU做时序建模。

    与 RL-TDAGCN 的区别: 有自适应图但没有RL控制器、拓扑约束、多模态融合。

    Args:
        input_dim:  输入特征维度
        hidden_dim: 隐藏层维度
        num_nodes:  节点数量
        embed_dim:  自适应图节点嵌入维度
        K:          扩散卷积阶数
        pred_len:   预测步数
    """

    def __init__(self, input_dim, hidden_dim, num_nodes,
                 embed_dim=10, K=2, pred_len=12):
        super().__init__()
        # 输入投影: F -> H
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # 自适应图生成器
        self.graph_gen = DynamicGraphGenerator(hidden_dim, embed_dim)
        # 扩散卷积层 (2层)
        self.diff_conv1 = DiffusionConv(hidden_dim, hidden_dim, K)
        self.diff_conv2 = DiffusionConv(hidden_dim, hidden_dim, K)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        # GRU 时序建模
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # 输出投影
        self.output_layer = nn.Linear(hidden_dim, pred_len)

    def forward(self, traffic, adj_static):
        """
        Args:
            traffic:    [B, T, N, F] 交通数据
            adj_static: [N, N]       静态邻接矩阵 (本模型不使用，保留接口统一)
        Returns:
            pred: [B, pred_len, N] 预测结果
            info: {} (空字典，统一接口)
        """
        B, T, N, Fea = traffic.shape

        # 输入投影: [B, T, N, H]
        h = self.input_proj(traffic)
        h = F.relu(h)

        # 生成自适应图: 对所有时间步取均值聚合 -> [B, N, H]
        h_mean = h.mean(dim=1)  # [B, N, H]
        A_adaptive = self.graph_gen(h_mean)  # [B, N, N]

        # 逐时间步做扩散卷积空间编码
        h_list = []
        for t in range(T):
            h_t = h[:, t, :, :]  # [B, N, H]
            # 第1层扩散卷积 + 残差 + LayerNorm
            residual = h_t
            h_t = self.diff_conv1(h_t, A_adaptive[0])  # 使用batch中第一个图的适配
            h_t = self.ln1(h_t + residual)
            h_t = F.relu(h_t)
            # 第2层扩散卷积 + 残差 + LayerNorm
            residual = h_t
            h_t = self.diff_conv2(h_t, A_adaptive[0])
            h_t = self.ln2(h_t + residual)
            h_t = F.relu(h_t)
            h_list.append(h_t)

        # 堆叠: [B, T, N, H]
        h = torch.stack(h_list, dim=1)

        # GRU 时序编码: [B*N, T, H] -> [B*N, T, H]
        h_flat = h.reshape(B * N, T, -1)
        h_flat, _ = self.gru(h_flat)
        h_flat = self.layer_norm(h_flat)
        h = h_flat.reshape(B, T, N, -1)

        # 取最后时间步预测
        h_last = h[:, -1, :, :]  # [B, N, H]
        pred = self.output_layer(h_last)  # [B, N, pred_len]
        pred = pred.permute(0, 2, 1)  # [B, pred_len, N]
        return pred, {}


# ======================================================================
# 3. STGCN (时空图卷积网络)
# 1层 DiffusionConv 做空间 + 1D Conv 做时序 + GLU激活
# ======================================================================
class STConvBlock(nn.Module):
    """
    时空卷积块
    结构: 时序1D Conv -> 空间DiffusionConv -> 时序1D Conv -> 残差连接

    Args:
        hidden_dim:  隐藏层维度
        K:           扩散卷积阶数
        kernel_size: 时序1D卷积核大小
        dropout:     Dropout率
    """

    def __init__(self, hidden_dim, K=2, kernel_size=3, dropout=0.3):
        super().__init__()
        # 时序卷积1 (带GLU门控)
        self.temporal_conv1 = nn.Conv1d(
            hidden_dim, 2 * hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        # 空间扩散卷积
        self.spatial_conv = DiffusionConv(hidden_dim, hidden_dim, K=K)
        self.ln_spatial = nn.LayerNorm(hidden_dim)
        # 时序卷积2 (带GLU门控)
        self.temporal_conv2 = nn.Conv1d(
            hidden_dim, 2 * hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )
        # 残差投影 (若维度变化)
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Args:
            x:   [B, T, N, H] 输入特征
            adj: [N, N]       邻接矩阵
        Returns:
            out: [B, T, N, H] 输出特征
        """
        B, T, N, H = x.shape
        residual = x

        # === 时序卷积1 + GLU门控 ===
        x_t = x.permute(0, 2, 3, 1).reshape(B * N, H, T)  # [B*N, H, T]
        x_t = self.temporal_conv1(x_t)  # [B*N, 2H, T]
        # GLU: 取前H维作为门控，后H维作为值
        x_t = x_t[:, :H, :] * torch.sigmoid(x_t[:, H:, :])  # [B*N, H, T]
        x = x_t.reshape(B, N, H, T).permute(0, 3, 1, 2)  # [B, T, N, H]
        x = self.dropout(x)

        # === 空间扩散卷积 ===
        h_list = []
        for t in range(T):
            h_t = self.spatial_conv(x[:, t, :, :], adj)  # [B, N, H]
            h_list.append(h_t)
        x = torch.stack(h_list, dim=1)  # [B, T, N, H]
        x = self.ln_spatial(x)
        x = F.relu(x)

        # === 时序卷积2 + GLU门控 ===
        x_t = x.permute(0, 2, 3, 1).reshape(B * N, H, T)  # [B*N, H, T]
        x_t = self.temporal_conv2(x_t)  # [B*N, 2H, T]
        x_t = x_t[:, :H, :] * torch.sigmoid(x_t[:, H:, :])  # [B*N, H, T]
        x = x_t.reshape(B, N, H, T).permute(0, 3, 1, 2)  # [B, T, N, H]
        x = self.dropout(x)

        # === 残差连接 ===
        x = x + residual
        return x


class STGCN(nn.Module):
    """
    STGCN 时空图卷积网络
    2个 STConvBlock 串联 + 输出层

    Args:
        input_dim:  输入特征维度
        hidden_dim: 隐藏层维度
        pred_len:   预测步数
        K:          扩散卷积阶数
    """

    def __init__(self, input_dim, hidden_dim, pred_len, K=2):
        super().__init__()
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # 2个时空卷积块
        self.st_block1 = STConvBlock(hidden_dim, K=K)
        self.st_block2 = STConvBlock(hidden_dim, K=K)
        # 最终层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # 输出投影
        self.output_layer = nn.Linear(hidden_dim, pred_len)

    def forward(self, traffic, adj_static):
        """
        Args:
            traffic:    [B, T, N, F] 交通数据
            adj_static: [N, N]       静态邻接矩阵
        Returns:
            pred: [B, pred_len, N] 预测结果
            info: {} (空字典，统一接口)
        """
        # 输入投影
        h = self.input_proj(traffic)  # [B, T, N, H]
        h = F.relu(h)

        # 两个时空卷积块
        h = self.st_block1(h, adj_static)
        h = self.st_block2(h, adj_static)

        # 层归一化
        h = self.layer_norm(h)

        # 取最后时间步预测
        h_last = h[:, -1, :, :]  # [B, N, H]
        pred = self.output_layer(h_last)  # [B, N, pred_len]
        pred = pred.permute(0, 2, 1)  # [B, pred_len, N]
        return pred, {}


# ======================================================================
# 4. TGCN (时序图卷积网络)
# GRU内部融合图卷积，将GRU的矩阵乘法替换为图卷积操作
# ======================================================================
class TGCNCell(nn.Module):
    """
    TGCN 单元: 融合了图卷积的GRU Cell
    将标准GRU中的矩阵乘法替换为图卷积操作，
    使门控和隐藏状态更新都考虑图结构信息。

    数学表达:
      update_gate = sigmoid(W_z * x + U_z * (A * h) + b_z)
      reset_gate  = sigmoid(W_r * x + U_r * (A * h) + b_r)
      h_tilde     = tanh(W * x + U * (A * (r * h)) + b)
      h_new       = (1 - update_gate) * h + update_gate * h_tilde

    注意: 图卷积这里用简单的 A @ h 实现（邻居聚合），
    而非完整的扩散卷积，与TGCN原始论文一致。

    Args:
        input_dim:  输入特征维度
        hidden_dim: 隐藏状态维度
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 更新门参数
        self.W_z = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))

        # 重置门参数
        self.W_r = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))

        # 候选隐藏状态参数
        self.W_h = nn.Linear(input_dim, hidden_dim, bias=False)
        self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h, adj):
        """
        单步前向传播

        Args:
            x:   [B, N, input_dim]  当前时刻输入
            h:   [B, N, hidden_dim] 上一时刻隐藏状态
            adj: [N, N]             邻接矩阵

        Returns:
            h_new: [B, N, hidden_dim] 新的隐藏状态
        """
        # 图卷积邻居聚合: A @ h
        h_graph = torch.matmul(adj, h)  # [B, N, hidden_dim]

        # 更新门
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_graph) + self.b_z)
        # 重置门
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_graph) + self.b_r)
        # 候选隐藏状态 (重置后的图卷积)
        h_reset_graph = torch.matmul(adj, r * h)  # [B, N, hidden_dim]
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(h_reset_graph) + self.b_h)
        # 最终隐藏状态
        h_new = (1.0 - z) * h + z * h_tilde
        return h_new


class TGCN(nn.Module):
    """
    TGCN 时序图卷积网络
    输入投影 -> 2层 TGCNCell 串联 -> 输出层

    Args:
        input_dim:  输入特征维度
        hidden_dim: 隐藏层维度
        pred_len:   预测步数
    """

    def __init__(self, input_dim, hidden_dim, pred_len):
        super().__init__()
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # 2层 TGCNCell
        self.tgcn_cell1 = TGCNCell(hidden_dim, hidden_dim)
        self.tgcn_cell2 = TGCNCell(hidden_dim, hidden_dim)
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # 输出投影
        self.output_layer = nn.Linear(hidden_dim, pred_len)

    def forward(self, traffic, adj_static):
        """
        Args:
            traffic:    [B, T, N, F] 交通数据
            adj_static: [N, N]       静态邻接矩阵
        Returns:
            pred: [B, pred_len, N] 预测结果
            info: {} (空字典，统一接口)
        """
        B, T, N, Fea = traffic.shape

        # 输入投影: [B, T, N, H]
        h = self.input_proj(traffic)
        h = F.relu(h)

        # 逐时间步通过TGCNCell
        # 第1层
        h1 = torch.zeros(B, N, self.tgcn_cell1.hidden_dim, device=traffic.device)
        h1_list = []
        for t in range(T):
            h1 = self.tgcn_cell1(h[:, t, :, :], h1, adj_static)
            h1_list.append(h1)

        # 第2层 (使用第1层的所有时间步输出)
        h2 = torch.zeros(B, N, self.tgcn_cell2.hidden_dim, device=traffic.device)
        for t in range(T):
            h2 = self.tgcn_cell2(h1_list[t], h2, adj_static)

        # 层归一化
        h2 = self.layer_norm(h2)

        # 输出预测
        pred = self.output_layer(h2)  # [B, N, pred_len]
        pred = pred.permute(0, 2, 1)  # [B, pred_len, N]
        return pred, {}


# ======================================================================
# 5. LSTM (纯时序基线)
# 没有图结构，纯LSTM处理每个节点
# ======================================================================
class LSTMModel(nn.Module):
    """
    LSTM 纯时序基线模型
    不使用任何图结构，将每个节点独立用LSTM处理。

    处理方式: 将 [B, T, N, F] reshape 为 [B*N, T, F] 送入LSTM，
    每个节点独立进行时序建模。

    Args:
        input_dim:  输入特征维度
        hidden_dim: LSTM隐藏维度
        pred_len:   预测步数
    """

    def __init__(self, input_dim, hidden_dim, pred_len):
        super().__init__()
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # 2层 LSTM (per-node)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # 输出投影
        self.output_layer = nn.Linear(hidden_dim, pred_len)

    def forward(self, traffic, adj_static):
        """
        Args:
            traffic:    [B, T, N, F] 交通数据
            adj_static: [N, N]       静态邻接矩阵 (本模型不使用，保留接口统一)
        Returns:
            pred: [B, pred_len, N] 预测结果
            info: {} (空字典，统一接口)
        """
        B, T, N, Fea = traffic.shape

        # 输入投影: [B, T, N, H]
        h = self.input_proj(traffic)
        h = F.relu(h)

        # 将 N 维度展平到 batch 维度: [B*N, T, H]
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)

        # LSTM 时序编码: [B*N, T, H]
        h, _ = self.lstm(h)

        # 层归一化
        h = self.layer_norm(h)

        # 取最后时间步: [B*N, H]
        h_last = h[:, -1, :]

        # 输出投影: [B*N, pred_len]
        pred = self.output_layer(h_last)

        # 恢复维度: [B, N, pred_len] -> [B, pred_len, N]
        pred = pred.reshape(B, N, -1)
        pred = pred.permute(0, 2, 1)  # [B, pred_len, N]
        return pred, {}