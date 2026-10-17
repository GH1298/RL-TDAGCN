"""
PEMS04 数据加载器
支持三特征 (flow, occupancy, speed) 和天气数据
数据划分: 70% 训练 / 15% 验证 / 15% 测试（硬约束）
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def load_adj_from_csv(csv_path, num_nodes, sigma=10.0):
    """
    从 PEMS04.csv (from, to, cost) 构建高斯核邻接矩阵 + 对称归一化
    
    Args:
        csv_path: 边距离CSV文件路径
        num_nodes: 节点数量
        sigma: 高斯核带宽参数
    
    Returns:
        adj_mx: (num_nodes, num_nodes) 归一化邻接矩阵
    """
    edges = pd.read_csv(csv_path)
    # 初始化距离矩阵为无穷大
    dist_mx = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
    
    for _, row in edges.iterrows():
        i, j = int(row['from']), int(row['to'])
        dist_mx[i][j] = row['cost']
        dist_mx[j][i] = row['cost']  # 对称化
    
    # 高斯核变换: W_ij = exp(-d_ij^2 / sigma^2)
    W = np.exp(-dist_mx ** 2 / (sigma ** 2))
    np.fill_diagonal(W, 1.0)  # 自环
    
    # 对称归一化: D^{-1/2} W D^{-1/2}
    d = np.sum(W, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    adj_mx = D_inv_sqrt @ W @ D_inv_sqrt
    
    return adj_mx.astype(np.float32)


class PEMS04Dataset(Dataset):
    """
    PEMS04 滑窗数据集
    
    Args:
        traffic: (T, N, F) 交通数据（已标准化）
        weather: (T, Fw) 天气数据（已标准化），可选
        seq_len: 输入序列长度
        pred_len: 预测步数
    """
    def __init__(self, traffic, weather=None, seq_len=12, pred_len=12):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        T, N, F = traffic.shape
        self.N = N
        self.F = F
        
        # 滑动窗口生成样本
        num_samples = T - seq_len - pred_len + 1
        self.traffic_x = np.zeros((num_samples, seq_len, N, F), dtype=np.float32)
        self.traffic_y = np.zeros((num_samples, pred_len, N), dtype=np.float32)  # 预测目标: 速度(第3列)
        
        for i in range(num_samples):
            self.traffic_x[i] = traffic[i:i + seq_len]
            # 预测目标取第3个特征（speed），shape: (pred_len, N)
            self.traffic_y[i] = traffic[i + seq_len:i + seq_len + pred_len, :, 2]
        
        # 天气数据（如果提供）
        self.has_weather = weather is not None
        if self.has_weather:
            self.weather = np.zeros((num_samples, seq_len, weather.shape[-1]), dtype=np.float32)
            for i in range(num_samples):
                self.weather[i] = weather[i:i + seq_len]
    
    def __len__(self):
        return len(self.traffic_x)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.traffic_x[idx])
        y = torch.FloatTensor(self.traffic_y[idx])
        if self.has_weather:
            w = torch.FloatTensor(self.weather[idx])
            return x, w, y
        return x, y


def create_data_loaders(data_dir, config):
    """
    创建训练/验证/测试 DataLoader
    
    Args:
        data_dir: PEMS04数据目录路径（包含 PEMS04_inner.npz 和 PEMS04.csv）
        config: 配置对象
    
    Returns:
        train_loader, val_loader, test_loader, adj_mx, scaler_stats
        scaler_stats: dict，包含 mean, std 用于反标准化
    """
    # 1. 加载交通数据
    npz_path = os.path.join(data_dir, config.npz_name)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"数据文件未找到: {npz_path}")
    
    raw = np.load(npz_path)
    if 'data' in raw:
        traffic = raw['data']  # (T, N, 3): flow, occupancy, speed
    else:
        traffic = raw[list(raw.keys())[0]]
    raw.close()
    
    print(f"[Data] 原始交通数据: {traffic.shape}")  # (16992, 307, 3)
    T = traffic.shape[0]
    
    # 2. 加载天气数据（可选）
    weather = None
    weather_path = os.path.join(data_dir, 'weather_aligned.npy')
    if config.use_multimodal and os.path.exists(weather_path):
        weather = np.load(weather_path)  # (T, Fw)
        print(f"[Data] 天气数据: {weather.shape}")
    elif config.use_multimodal:
        print(f"[Data] 警告: 多模态已开启但未找到天气文件 {weather_path}")
    
    # 3. 数据划分: 70% / 15% / 15%
    train_end = int(T * 0.70)
    val_end = int(T * 0.85)
    
    train_traffic = traffic[:train_end]
    val_traffic = traffic[train_end:val_end]
    test_traffic = traffic[val_end:]
    
    # 4. Z-Score 标准化（仅在训练集上拟合）
    mean = train_traffic.mean(axis=0, keepdims=True)  # (1, N, 3)
    std = train_traffic.std(axis=0, keepdims=True) + 1e-8  # (1, N, 3)
    
    train_traffic = (train_traffic - mean) / std
    val_traffic = (val_traffic - mean) / std
    test_traffic = (test_traffic - mean) / std
    
    # 天气标准化
    if weather is not None:
        w_mean = weather[:train_end].mean(axis=0, keepdims=True)
        w_std = weather[:train_end].std(axis=0, keepdims=True) + 1e-8
        weather = (weather - w_mean) / w_std
        
        train_weather = weather[:train_end]
        val_weather = weather[train_end:val_end]
        test_weather = weather[val_end:]
    else:
        train_weather = val_weather = test_weather = None
    
    scaler_stats = {'mean': mean, 'std': std}
    if weather is not None:
        scaler_stats['w_mean'] = w_mean
        scaler_stats['w_std'] = w_std
    
    # 5. 构建邻接矩阵
    csv_path = os.path.join(data_dir, config.csv_name)
    adj_mx = load_adj_from_csv(csv_path, config.num_nodes)
    print(f"[Data] 邻接矩阵: {adj_mx.shape}, 非零: {np.count_nonzero(adj_mx)}")
    
    # 6. 创建数据集
    train_dataset = PEMS04Dataset(train_traffic, train_weather, config.seq_len, config.pred_len)
    val_dataset = PEMS04Dataset(val_traffic, val_weather, config.seq_len, config.pred_len)
    test_dataset = PEMS04Dataset(test_traffic, test_weather, config.seq_len, config.pred_len)
    
    print(f"[Data] 样本数 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")
    
    # 7. 创建 DataLoader
    # [v4优化] 添加 num_workers 和 pin_memory 加速数据加载
    # num_workers=4: 4个子进程并行预取数据, 减少CPU-GPU等待
    # pin_memory=True: 锁页内存, 加速 CPU->GPU 数据传输 (仅CUDA生效)
    # [v4新增] 根据设备类型自动选择是否启用pin_memory
    use_cuda = hasattr(config, 'device') and 'cuda' in str(getattr(config, 'device', 'cpu'))
    num_workers = 4 if use_cuda else 0  # CPU模式不用多进程
    pin_memory = use_cuda

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=(num_workers > 0))
    # 注意: 测试集用 batch_size=1 方便逐样本评估
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, drop_last=False,
                             num_workers=0, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, adj_mx, scaler_stats


# 兼容性别名，方便旧代码或导入使用
PEMS04Loader = PEMS04Dataset