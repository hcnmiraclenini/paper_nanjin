"""
MoE-Nanjin 工具函数
"""

import torch
import numpy as np


def mape_with_threshold(y_true, y_pred, threshold=10.0):
    """计算MAPE（带阈值过滤，如果阈值过滤后无样本则使用自适应阈值）"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # 裁剪预测值为非负
    y_pred = np.clip(y_pred, 0, None)
    
    # 首先尝试使用原始阈值
    mask = (y_true > threshold) & (y_pred > threshold)
    
    # 如果原始阈值下没有有效样本，尝试使用更低的阈值
    if mask.sum() == 0:
        # 使用数据的中位数作为自适应阈值
        adaptive_threshold = max(np.median(y_true[y_true > 0]), 1.0)  # 至少为1.0
        mask = (y_true > adaptive_threshold) & (y_pred > adaptive_threshold)
        
        # 如果仍然没有有效样本，使用所有非零样本
        if mask.sum() == 0:
            mask = (y_true > 0) & (y_pred > 0)
            if mask.sum() == 0:
                # 如果完全没有有效样本，返回一个默认值（表示无法计算）
                return None, 0, len(y_true)
    
    mape = np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-8)).mean() * 100
    return mape, mask.sum(), len(y_true) - mask.sum()


def mse_loss(y_true, y_pred, eps=1e-8):
    """
    计算归一化MSE（按照实验要求）
    MSE = Σ(yi - ŷi)² / Σyi²
    """
    if isinstance(y_true, torch.Tensor):
        errors_squared = (y_true - y_pred) ** 2  # [B, output_dim] 或 [N, output_dim]
        sum_errors_squared = errors_squared.sum()  # 所有误差的平方和
        sum_true_squared = (y_true ** 2).sum()  # 所有真实值的平方和
        return sum_errors_squared / (sum_true_squared + eps)
    else:
        errors_squared = (y_true - y_pred) ** 2
        sum_errors_squared = np.sum(errors_squared)
        sum_true_squared = np.sum(y_true ** 2)
        return sum_errors_squared / (sum_true_squared + eps)


def mae_loss(y_true, y_pred, eps=1e-8):
    """
    计算归一化MAE（修复版本，使用绝对值之和作为分母以避免归一化数据的不稳定性）
    MAE = Σ|yi - ŷi| / Σ|yi|
    
    注意：原公式 MAE = Σ|yi - ŷi| / Σyi 在归一化数据上不稳定（y_true的和可能接近0或为负）
    """
    if isinstance(y_true, torch.Tensor):
        abs_errors = torch.abs(y_true - y_pred)  # [B, output_dim] 或 [N, output_dim]
        sum_abs_errors = abs_errors.sum()  # 所有绝对误差的和
        sum_abs_true = y_true.abs().sum()  # 所有真实值的绝对值之和（避免负分母）
        return sum_abs_errors / (sum_abs_true + eps)
    else:
        abs_errors = np.abs(y_true - y_pred)
        sum_abs_errors = np.sum(abs_errors)
        sum_abs_true = np.sum(np.abs(y_true))
        return sum_abs_errors / (sum_abs_true + eps)


def mape_loss(y_true, y_pred, threshold=10.0, eps=1e-8):
    """计算MAPE损失（用于训练）"""
    if isinstance(y_true, torch.Tensor):
        y_pred = torch.clamp(y_pred, min=0)
        mask = (y_true > threshold) & (y_pred > threshold)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y_true.device)
        return torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100
    else:
        y_pred = np.clip(y_pred, 0, None)
        mask = (y_true > threshold) & (y_pred > threshold)
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100


def weighted_mape_loss(y_true, y_pred, variable_weights=None, threshold=10.0, eps=1e-8):
    """
    加权MAPE损失：为不同变量分配不同权重（小流量站点权重更高）
    
    Args:
        y_true: [B, output_dim] 真实值
        y_pred: [B, output_dim] 预测值
        variable_weights: [output_dim] 每个变量的权重，如果None则使用自适应权重
        threshold: 阈值
        eps: 小常数避免除零
    """
    if isinstance(y_true, torch.Tensor):
        y_pred = torch.clamp(y_pred, min=0)
        mask = (y_true > threshold) & (y_pred > threshold)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y_true.device)
        
        # 计算每个位置的MAPE [B, output_dim]
        mape_per_element = torch.abs((y_true - y_pred) / (y_true + eps)) * 100
        
        # 如果没有提供权重，使用自适应权重（小流量站点权重更高）
        if variable_weights is None:
            # 根据真实值的均值计算权重：均值越小，权重越大
            if len(y_true.shape) == 2:
                # [B, output_dim] - 假设output_dim = n_vars * horizon
                # 计算每个位置的均值
                var_means = y_true.mean(dim=0)  # [output_dim]
                # 权重与均值成反比，归一化
                weights = 1.0 / (var_means + eps)
                weights = weights / weights.mean()  # 归一化，使平均权重为1
                variable_weights = weights
            else:
                variable_weights = torch.ones(y_true.shape[-1], device=y_true.device)
        
        # 扩展权重到batch维度
        if len(y_true.shape) == 2:
            weights_expanded = variable_weights.unsqueeze(0).expand_as(y_true)  # [B, output_dim]
            # 只对有效位置应用权重
            weighted_mape_elements = mape_per_element * weights_expanded * mask.float()
            weighted_mape = weighted_mape_elements.sum() / mask.float().sum()
        else:
            weighted_mape = mape_per_element[mask].mean()
        
        return weighted_mape
    else:
        # numpy版本
        y_pred = np.clip(y_pred, 0, None)
        mask = (y_true > threshold) & (y_pred > threshold)
        if mask.sum() == 0:
            return 0.0
        
        mape_per_element = np.abs((y_true - y_pred) / (y_true + eps)) * 100
        
        if variable_weights is None:
            if len(y_true.shape) == 2:
                var_means = y_true.mean(axis=0)
                weights = 1.0 / (var_means + eps)
                weights = weights / weights.mean()
                variable_weights = weights
            else:
                variable_weights = np.ones(y_true.shape[-1])
        
        if len(y_true.shape) == 2:
            weights_expanded = np.broadcast_to(variable_weights, y_true.shape)
            weighted_mape_elements = mape_per_element * weights_expanded * mask.astype(float)
            weighted_mape = weighted_mape_elements.sum() / mask.astype(float).sum()
        else:
            weighted_mape = mape_per_element[mask].mean()
        
        return weighted_mape

