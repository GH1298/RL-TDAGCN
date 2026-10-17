"""
评估指标
包含: MAE(主), RMSE, SMAPE(对称MAPE，替代MAPE以应对零值问题)
"""
import torch
import numpy as np


def masked_mae(pred, true, threshold=1e-3):
    """带掩码的 MAE（主指标）"""
    mask = (torch.abs(true) >= threshold).float()
    mask_sum = mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0)
    loss = torch.abs(pred - true) * mask
    return loss.sum() / mask_sum


def masked_rmse(pred, true, threshold=1e-3):
    """带掩码的 RMSE"""
    mask = (torch.abs(true) >= threshold).float()
    mask_sum = mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0)
    loss = (pred - true) ** 2 * mask
    return torch.sqrt(loss.sum() / mask_sum)


def masked_smape(pred, true, threshold=1e-3):
    """
    对称平均绝对百分比误差 (SMAPE)
    SMAPE = 200% * |pred - true| / (|pred| + |true| + eps)
    比MAPE更适合含零值的交通数据
    """
    mask = (torch.abs(true) >= threshold).float()
    mask_sum = mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0)
    numerator = torch.abs(pred - true)
    denominator = (torch.abs(pred) + torch.abs(true)) + 1e-8
    smape = 2.0 * numerator / denominator
    return (smape * mask).sum() / mask_sum


def masked_mae_np(pred, true, threshold=1e-3):
    """NumPy 版本 MAE"""
    mask = (np.abs(true) >= threshold)
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(pred - true) * mask)


def masked_rmse_np(pred, true, threshold=1e-3):
    """NumPy 版本 RMSE"""
    mask = (np.abs(true) >= threshold)
    if mask.sum() == 0:
        return 0.0
    return np.sqrt(np.mean((pred - true) ** 2 * mask))


def masked_smape_np(pred, true, threshold=1e-3):
    """NumPy 版本 SMAPE"""
    mask = (np.abs(true) >= threshold)
    if mask.sum() == 0:
        return 0.0
    numerator = np.abs(pred - true)
    denominator = np.abs(pred) + np.abs(true) + 1e-8
    return np.mean(2.0 * numerator / denominator * mask)


def compute_all_metrics(pred, true):
    """
    计算所有指标
    
    Args:
        pred: (B, H, N) 预测值（反标准化后）
        true: (B, H, N) 真实值（反标准化后）
    
    Returns:
        dict: MAE, RMSE, SMAPE
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    
    return {
        'MAE': masked_mae_np(pred, true),
        'RMSE': masked_rmse_np(pred, true),
        'SMAPE': masked_smape_np(pred, true),
    }