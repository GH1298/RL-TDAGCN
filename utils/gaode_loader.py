"""
高德数据加载器
==============
读取高德交通数据采集器生成的 numpy 数据，创建与 PEMS04 接口一致的 DataLoader。

与 utils/dataloader.py 中 create_data_loaders() 的区别:
    1. 读取 traffic.npy 而非 PEMS04_inner.npz
    2. 读取 adj.csv (from, to, cost) 格式
    3. 自动从 meta.json 检测节点数(无需手动指定 num_nodes)
    4. 自动检测 weather_aligned.npy 是否存在

复用的现有逻辑:
    - PEMS04Dataset (滑动窗口数据集类)
    - load_adj_from_csv (高斯核邻接矩阵构建)

数据划分与标准化策略与 PEMS04 完全一致:
    - 70% 训练 / 15% 验证 / 15% 测试
    - Z-Score 标准化(仅在训练集上拟合)
"""
import os
import json
import numpy as np
import torch

# 复用现有 dataloader 中的数据集类和邻接矩阵构建函数
# 这样保证数据处理逻辑完全一致，只改数据来源
from .dataloader import PEMS04Dataset, load_adj_from_csv
from torch.utils.data import DataLoader


def _load_meta(data_dir):
    """
    从 meta.json 加载采集元信息

    Args:
        data_dir: 数据目录

    Returns:
        dict: 包含 num_nodes 等元信息；若文件不存在返回空 dict
    """
    meta_path = os.path.join(data_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _detect_num_nodes(data_dir, traffic, config):
    """
    自动检测节点数

    检测优先级:
        1. config.num_nodes (显式指定)
        2. meta.json 中的 num_nodes
        3. traffic.npy 的第二维(N)

    Args:
        data_dir: 数据目录
        traffic: (T, N, F) 交通数据
        config: 配置对象

    Returns:
        num_nodes: 节点数量
    """
    # 优先用数据实际形状
    num_nodes = traffic.shape[1]
    meta = _load_meta(data_dir)
    if 'num_nodes' in meta and meta['num_nodes'] != num_nodes:
        print(f'[GaodeLoader] 警告: meta.json 节点数({meta["num_nodes"]}) '
              f'与 traffic.npy({num_nodes}) 不一致，以数据为准')
    # 若 config 显式指定且不匹配，给出提示但仍以数据为准
    if hasattr(config, 'num_nodes') and config.num_nodes != num_nodes:
        print(f'[GaodeLoader] 提示: config.num_nodes={config.num_nodes}, '
              f'实际数据节点数={num_nodes}，已自动适配')
    return num_nodes


def create_gaode_data_loaders(data_dir, config):
    """
    创建高德数据的 DataLoader

    与 PEMS04 的 create_data_loaders 接口完全一致，返回相同结构的对象，
    因此 train.py 无需修改即可直接切换到高德数据训练。

    接口约定:
        输入: data_dir (高德数据目录), config
        输出: train_loader, val_loader, test_loader, adj_mx, scaler_stats

    与 PEMS04 版本的差异:
        1. 交通数据来源: traffic.npy (而非 PEMS04_inner.npz)
        2. 邻接矩阵来源: adj.csv (而非 PEMS04.csv)
        3. 节点数自动检测 (而非固定 307)
        4. 天气数据: weather_aligned.npy (高德采集的真实天气)

    Args:
        data_dir: 高德数据目录路径，需包含:
            - traffic.npy: (T, N, 3) 交通数据
            - adj.csv: (from, to, cost) 边列表
            - meta.json: 元信息(可选)
            - weather_aligned.npy: 天气数据(可选)
        config: 配置对象，至少需要:
            - seq_len, pred_len, batch_size
            - use_multimodal (是否启用天气)
            - sigma (高斯核带宽，可选，默认10.0)

    Returns:
        train_loader, val_loader, test_loader, adj_mx, scaler_stats
        scaler_stats: dict，包含 mean, std (以及 w_mean, w_std 若有天气)
    """
    # 1. 加载交通数据 (traffic.npy)
    traffic_path = os.path.join(data_dir, 'traffic.npy')
    if not os.path.exists(traffic_path):
        raise FileNotFoundError(
            f'交通数据文件未找到: {traffic_path}\n'
            f'请先运行 utils/gaode_fetcher.py 采集并处理数据。'
        )
    traffic = np.load(traffic_path)  # (T, N, 3): flow, occupancy, speed
    print(f'[GaodeLoader] 原始交通数据: {traffic.shape}')  # 例如 (4032, 156, 3)
    T = traffic.shape[0]

    # 2. 自动检测节点数
    num_nodes = _detect_num_nodes(data_dir, traffic, config)
    print(f'[GaodeLoader] 节点数(道路数): {num_nodes}')

    # 3. 加载天气数据(可选)
    weather = None
    weather_path = os.path.join(data_dir, 'weather_aligned.npy')
    if getattr(config, 'use_multimodal', False) and os.path.exists(weather_path):
        weather = np.load(weather_path)  # (T, 6)
        print(f'[GaodeLoader] 天气数据: {weather.shape}')
        # 天气时间步数应与交通一致，不一致则截断对齐
        if weather.shape[0] != T:
            print(f'[GaodeLoader] 警告: 天气时间步({weather.shape[0]}) '
                  f'与交通({T})不一致，自动截断对齐')
            min_t = min(weather.shape[0], T)
            weather = weather[:min_t]
            traffic = traffic[:min_t]
            T = min_t
    elif getattr(config, 'use_multimodal', False):
        print(f'[GaodeLoader] 警告: 多模态已开启但未找到天气文件 {weather_path}')

    # 4. 数据划分: 70% / 15% / 15% (与 PEMS04 完全一致)
    train_end = int(T * 0.70)
    val_end = int(T * 0.85)

    train_traffic = traffic[:train_end]
    val_traffic = traffic[train_end:val_end]
    test_traffic = traffic[val_end:]

    # 5. Z-Score 标准化(仅在训练集上拟合，与 PEMS04 一致)
    mean = train_traffic.mean(axis=0, keepdims=True)  # (1, N, 3)
    std = train_traffic.std(axis=0, keepdims=True) + 1e-8  # (1, N, 3)

    train_traffic = (train_traffic - mean) / std
    val_traffic = (val_traffic - mean) / std
    test_traffic = (test_traffic - mean) / std

    # 天气标准化(与 PEMS04 一致)
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

    # 6. 构建邻接矩阵 (复用 load_adj_from_csv，与 PEMS04 完全一致)
    adj_path = os.path.join(data_dir, 'adj.csv')
    if not os.path.exists(adj_path):
        raise FileNotFoundError(
            f'邻接边文件未找到: {adj_path}\n'
            f'请先运行 utils/gaode_fetcher.py 处理数据。'
        )
    sigma = getattr(config, 'sigma', 10.0)
    adj_mx = load_adj_from_csv(adj_path, num_nodes, sigma=sigma)
    print(f'[GaodeLoader] 邻接矩阵: {adj_mx.shape}, 非零: {np.count_nonzero(adj_mx)}')

    # 7. 创建数据集 (复用 PEMS04Dataset)
    train_dataset = PEMS04Dataset(train_traffic, train_weather,
                                  config.seq_len, config.pred_len)
    val_dataset = PEMS04Dataset(val_traffic, val_weather,
                                config.seq_len, config.pred_len)
    test_dataset = PEMS04Dataset(test_traffic, test_weather,
                                config.seq_len, config.pred_len)

    print(f'[GaodeLoader] 样本数 - 训练: {len(train_dataset)}, '
          f'验证: {len(val_dataset)}, 测试: {len(test_dataset)}')

    # 8. 创建 DataLoader (与 PEMS04 一致)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, adj_mx, scaler_stats
