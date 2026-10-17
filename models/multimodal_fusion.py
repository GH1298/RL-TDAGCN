"""
多模态信息融合模块 (Multimodal Fusion)
包含:
  1. WeatherEncoder: 天气特征编码器
  2. GatingFusion: 门控融合机制

设计思路:
  将天气等外部信息融合到交通特征中。
  使用门控机制让模型自适应决定不同模态的融合权重。

数学表达:
  w_feat = MLP(weather)                    # 天气特征编码
  gate = sigmoid(W * [t_feat; w_feat] + b) # 门控权重
  fused = gate * t_feat + (1-gate) * w_feat_broadcast  # 门控融合
"""
import torch
import torch.nn as nn


class WeatherEncoder(nn.Module):
    """
    天气编码器

    将原始天气特征编码到与交通特征相同的隐藏空间

    Args:
        weather_dim: 天气特征维度 (如6: 温度/湿度/风速/气压/降水/能见度)
        hidden_dim: 输出隐藏维度
    """
    def __init__(self, weather_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, weather):
        """
        Args:
            weather: [B, T, Fw] 天气特征

        Returns:
            w_feat: [B, T, hidden_dim] 天气编码特征
        """
        return self.encoder(weather)


class GatingFusion(nn.Module):
    """
    门控融合模块

    自适应融合交通特征和天气特征。
    门控权重由两种特征联合决定。

    Args:
        hidden_dim: 特征维度
        gate_hidden: 门控网络隐藏维度
    """
    def __init__(self, hidden_dim, gate_hidden=32):
        super().__init__()
        # 门控网络: 输入为 [t_feat; w_feat] 拼接
        # t_feat: [B, T, N, H], w_feat: [B, T, H] -> 需要广播对齐
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, traffic_feat, weather_feat):
        """
        门控融合

        Args:
            traffic_feat: [B, T, N, H] 交通编码特征
            weather_feat: [B, T, H] 天气编码特征

        Returns:
            fused: [B, T, N, H] 融合后的特征
        """
        B, T, N, H = traffic_feat.shape
        Wf = weather_feat.shape[-1]

        # 将天气特征广播到节点维度: [B, T, 1, Wf] -> [B, T, N, Wf]
        w_expanded = weather_feat.unsqueeze(2).expand(B, T, N, Wf)

        # 如果天气特征维度 != 交通特征维度，需要投影
        if Wf != H:
            # 这个情况应该在上游处理，这里做防御
            # 简单截断或补零
            if Wf > H:
                w_expanded = w_expanded[:, :, :, :H]
            else:
                pad = torch.zeros(B, T, N, H - Wf, device=traffic_feat.device)
                w_expanded = torch.cat([w_expanded, pad], dim=-1)

        # 拼接: [B, T, N, 2H]
        combined = torch.cat([traffic_feat, w_expanded], dim=-1)

        # 计算门控权重: [B, T, N, 1]
        gate = self.gate_net(combined)

        # 门控融合: gate * traffic + (1-gate) * weather
        fused = gate * traffic_feat + (1 - gate) * w_expanded

        return fused