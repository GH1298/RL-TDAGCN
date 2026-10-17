"""
动态图生成器 (Dynamic Graph Generator)
基于 GraphWaveNet 的思想，通过两层 MLP 学习节点嵌入，
再通过注意力机制生成动态邻接矩阵 A_t = softmax(node_embed_i^T * node_embed_j / tau)

数学表达:
  E = MLP_2(ReLU(MLP_1(X)))    # 节点嵌入, [B, N, d]
  A_t = softmax(E * E^T / tau)  # 动态邻接矩阵, [B, N, N]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEmbedding(nn.Module):
    """
    节点嵌入模块
    将节点特征映射到低维嵌入空间
    
    Args:
        input_dim: 输入特征维度 (hidden_dim, 来自编码器)
        embed_dim: 嵌入维度
    """
    def __init__(self, input_dim, embed_dim=10):
        super().__init__()
        self.embed_dim = embed_dim
        # 两层 MLP 映射
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N, input_dim] 节点特征
        Returns:
            embed: [B, N, embed_dim] 节点嵌入
        """
        return self.mlp(x)


class DynamicGraphGenerator(nn.Module):
    """
    动态图生成器
    
    根据当前时刻的节点特征，生成自适应邻接矩阵。
    使用多头注意力机制提高图学习表达能力。
    
    Args:
        input_dim: 节点特征维度
        embed_dim: 节点嵌入维度
        num_heads: 注意力头数（多头图注意力）
        temperature: softmax 温度参数，控制邻接矩阵的锐度
    """
    def __init__(self, input_dim, embed_dim=10, num_heads=4, temperature=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        self.embed_dim = embed_dim
        
        # 每个注意力头独立的节点嵌入
        self.head_embeddings = nn.ModuleList([
            NodeEmbedding(input_dim, embed_dim) for _ in range(num_heads)
        ])
        
        # 输出投影：将多头结果融合为单个邻接矩阵
        self.output_proj = nn.Sequential(
            nn.Linear(num_heads, 1),
            nn.Sigmoid()  # 保证输出在 [0, 1] 范围
        )
    
    def forward(self, node_features):
        """
        生成动态邻接矩阵
        
        Args:
            node_features: [B, N, input_dim] 来自编码器的节点特征
        
        Returns:
            adj: [B, N, N] 动态邻接矩阵（每行softmax归一化）
        """
        B, N, _ = node_features.shape
        
        # 多头注意力计算
        head_adjs = []
        for head in self.head_embeddings:
            embed = head(node_features)  # [B, N, embed_dim]
            # 计算节点相似度矩阵
            sim = torch.bmm(embed, embed.transpose(1, 2))  # [B, N, N]
            # softmax 归一化（每行），温度缩放
            adj = F.softmax(sim / self.temperature, dim=-1)  # [B, N, N]
            head_adjs.append(adj)
        
        # 堆叠多头结果: [B, N, N, num_heads]
        stacked = torch.stack(head_adjs, dim=-1)
        # 融合为单个邻接矩阵
        adj = self.output_proj(stacked).squeeze(-1)  # [B, N, N]
        
        # 对称化: A = (A + A^T) / 2
        adj = (adj + adj.transpose(1, 2)) / 2.0
        
        return adj
    
    def get_candidate_graph(self, node_features):
        """
        获取候选动态图（与forward相同，语义更明确）
        RL控制器会基于此候选图决定最终图
        """
        return self.forward(node_features)