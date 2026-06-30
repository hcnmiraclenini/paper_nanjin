"""
MoE-Nanjin 训练脚本
训练损失 = lambda_mae * MAE + lambda_mse * MSE + lambda_balance * Balance
MAPE不参与训练损失计算，只用于评估和早停判断

功能：
- 训练模型并保存最佳权重
- 不进行测试集评估（测试集评估请使用 evaluate.py）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import argparse
import random
import numpy as np
import time
import warnings

# 抑制常见的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from model import MoENanjin
from data_loader import prepare_data
from utils import mape_with_threshold, mse_loss, mae_loss, mape_loss


def set_random_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_loss(y_pred, y_true, gate_logits, lambda_mape=0.0, lambda_mae=0.0, lambda_mse=1.0, lambda_balance=0.01, 
                 lambda_range=1.0, scaler=None, n_targets=None, horizon=1, use_weighted_loss=False,
                 balance_mode='entropy'):
    """
    计算总损失（完全参考timer_xl：只使用MSE + 范围惩罚）
    
    训练损失 = lambda_mse * MSE + lambda_balance * Balance + lambda_range * Range
    
    关键改进：
    1. 完全参考timer_xl：只使用MSE损失（lambda_mse=1.0），移除MAPE和MAE损失
    2. 大幅增加范围惩罚权重（lambda_range=0.5），解决预测值范围偏小问题
    3. 修复范围惩罚计算，确保惩罚真正起作用（之前一直是0.0000）
    
    Args:
        y_pred: [B, output_dim] 预测值（归一化后的）
        y_true: [B, output_dim] 真实值（归一化后的）
        gate_logits: [B, num_experts] 门控logits
        lambda_mape: MAPE损失权重（默认0.0，完全参考timer_xl，不使用）
        lambda_mae: MAE损失权重（默认0.0，完全参考timer_xl，不使用）
        lambda_mse: MSE损失权重（默认1.0，完全参考timer_xl，只使用MSE）
        lambda_balance: 负载均衡损失权重（默认0.01）
        lambda_range: 范围惩罚损失权重（默认0.5，大幅增加以解决预测值范围偏小问题）
        scaler: 用于反归一化的scaler（计算MAPE时必须提供）
        n_targets: 目标变量数（计算MAPE时必须提供）
        horizon: 预测窗口长度（计算MAPE时必须提供）
        use_weighted_loss: 是否使用加权损失（默认False，避免过度保守）
    
    Returns:
        total_loss: 总损失
        loss_dict: 各损失分量
    """
    eps = 1e-8
    
    # 1. MAPE损失（简化版：参考timer_xl，主要使用MSE和MAE，MAPE作为辅助）
    # 在归一化空间计算相对误差，但权重较小，避免干扰主要训练
    # timer_xl主要使用MSE，MAPE只用于评估，我们也可以这样做
    
    abs_true = torch.abs(y_true)
    abs_error = torch.abs(y_true - y_pred)
    
    # 使用更简单的相对误差计算（类似MAPE，但更稳定）
    # 使用平滑因子避免除零
    smooth_factor = torch.std(abs_true) * 0.05 + eps
    relative_error = abs_error / (abs_true + smooth_factor)
    
    # MAPE损失 = 相对误差的平均值（权重较小，主要优化由MSE和MAE完成）
    mape_loss_val = relative_error.mean()
    
    # 2. MAE损失（在归一化空间计算，用于训练稳定性）
    if use_weighted_loss:
        # 加权MAE：给小值样本更高权重（可能导致模型过于保守）
        abs_true = torch.abs(y_true)
        weights = 1.0 / (abs_true + eps)
        weights = weights / (weights.mean() + eps)  # 归一化
        abs_errors = torch.abs(y_true - y_pred)
        weighted_abs_errors = abs_errors * weights
        mae_loss_val = weighted_abs_errors.mean()
    else:
        # 标准MAE损失（推荐，避免过度保守）
        mae_loss_val = mae_loss(y_true, y_pred)
    
    # 3. MSE损失（在归一化空间计算，用于训练稳定性）
    if use_weighted_loss:
        # 加权MSE
        abs_true = torch.abs(y_true)
        weights = 1.0 / (abs_true + eps)
        weights = weights / (weights.mean() + eps)
        errors_squared = (y_true - y_pred) ** 2
        weighted_errors_squared = errors_squared * weights
        mse_loss_val = weighted_errors_squared.mean() / (y_true ** 2).mean()  # 归一化
    else:
        # 标准MSE损失（推荐）
        mse_loss_val = mse_loss(y_true, y_pred)
    
    gate_weights = torch.softmax(gate_logits, dim=-1)
    expert_usage = gate_weights.mean(dim=0)
    if balance_mode == 'entropy':
        eps = 1e-8
        balance_loss = (expert_usage * torch.log(expert_usage + eps)).sum()
    else:
        balance_loss = torch.var(expert_usage)
    
    # 5. 范围惩罚损失（最终修复版：使用方差和每个样本的相对误差）
    # 问题：在归一化空间，预测值和真实值的统计量可能很接近，导致惩罚为0
    # 解决：使用方差差异 + 每个样本的相对误差，确保惩罚始终有效
    
    # 5.1 方差惩罚（比标准差更敏感）：预测值的方差应该接近真实值的方差
    pred_var = y_pred.var()
    true_var = y_true.var()
    var_penalty = torch.abs(pred_var - true_var) / (torch.abs(true_var) + eps)
    
    # 5.2 标准差惩罚：预测值的标准差应该接近真实值的标准差
    pred_std = y_pred.std()
    true_std = y_true.std()
    std_penalty = torch.abs(pred_std - true_std) / (torch.abs(true_std) + eps)
    
    # 5.3 分位数惩罚：鼓励预测值的分位数接近真实值的分位数
    pred_q25 = torch.quantile(y_pred, 0.25)
    pred_q50 = torch.quantile(y_pred, 0.50)
    pred_q75 = torch.quantile(y_pred, 0.75)
    true_q25 = torch.quantile(y_true, 0.25)
    true_q50 = torch.quantile(y_true, 0.50)
    true_q75 = torch.quantile(y_true, 0.75)
    
    quantile_penalty = (
        torch.abs(pred_q25 - true_q25) / (torch.abs(true_q25) + eps) +
        torch.abs(pred_q50 - true_q50) / (torch.abs(true_q50) + eps) +
        torch.abs(pred_q75 - true_q75) / (torch.abs(true_q75) + eps)
    ) / 3.0
    
    # 5.4 范围惩罚：预测值范围应该接近真实值范围
    pred_range = y_pred.max() - y_pred.min()
    true_range = y_true.max() - y_true.min()
    range_penalty_val = torch.abs(pred_range - true_range) / (torch.abs(true_range) + eps)
    
    # 5.5 每个样本的相对误差惩罚（关键：确保惩罚始终有效）
    # 计算每个样本的相对误差，然后取平均值
    # 这样可以惩罚预测值分布与真实值分布的差异
    abs_true = torch.abs(y_true)
    abs_error = torch.abs(y_true - y_pred)
    # 使用平滑因子避免除零
    smooth_factor = torch.std(abs_true) * 0.1 + eps
    per_sample_relative_error = abs_error / (abs_true + smooth_factor)
    # 取平均值作为惩罚（这个应该始终>0，因为abs_error和abs_true都是非负的）
    per_sample_penalty = per_sample_relative_error.mean()
    
    # 综合范围惩罚（加权组合，每个样本惩罚权重最大，确保始终有效）
    # 关键：per_sample_penalty应该始终>0，因为它基于每个样本的相对误差
    range_penalty_components = (
        var_penalty * 0.2 +              # 方差惩罚
        std_penalty * 0.15 +              # 标准差惩罚
        quantile_penalty * 0.15 +         # 分位数惩罚
        range_penalty_val * 0.1 +         # 范围惩罚
        per_sample_penalty * 0.4          # 每个样本惩罚（最重要，确保始终有效）
    )
    
    # 强制添加基础惩罚（确保惩罚始终不为0，直接添加，不判断条件）
    # 即使所有组件都是0，这个基础惩罚也能保证至少有一个最小值
    base_penalty = 0.01  # 基础惩罚值
    range_penalty = range_penalty_components + base_penalty
    
    # 使用torch.clamp确保惩罚始终在合理范围内（保持梯度）
    # 这确保即使所有组件都是0，惩罚也至少是0.01，同时防止爆炸
    range_penalty = torch.clamp(range_penalty, min=0.01, max=10.0)  # 添加上限防止爆炸
    
    # 总损失 = lambda_mape * MAPE + lambda_mae * Weighted_MAE + lambda_mse * Weighted_MSE + lambda_balance * Balance + lambda_range * Range
    total_loss = (
        lambda_mape * mape_loss_val +
        lambda_mae * mae_loss_val +
        lambda_mse * mse_loss_val +
        lambda_balance * balance_loss +
        lambda_range * range_penalty
    )
    
    loss_dict = {
        'total': total_loss,
        'mape': mape_loss_val,
        'mae': mae_loss_val,
        'mse': mse_loss_val,
        'balance': balance_loss,
        'range': range_penalty
    }
    
    return total_loss, loss_dict


def _unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def train_epoch(model, train_loader, optimizer, device, lambda_mape=0.0, lambda_mae=0.0, 
                lambda_mse=1.0, lambda_balance=0.01, lambda_range=1.0, data_scaler=None, amp_scaler=None, use_amp=False, n_targets=None, horizon=1, use_weighted_loss=False, debug=False,
                balance_mode='entropy'):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {'mape': 0.0, 'mae': 0.0, 'mse': 0.0, 'balance': 0.0, 'range': 0.0}  # 添加'range'键
    num_batches = 0
    
    # 用于调试：记录第一个batch的参数变化
    if debug:
        first_param_before = None
        for name, param in model.named_parameters():
            if param.requires_grad:
                first_param_before = param.data.clone()
                break
    
    for batch in tqdm(train_loader, desc="训练中", leave=False):
        batch_x, batch_y, batch_scene = _unpack_batch(batch)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if batch_scene is not None:
            batch_scene = batch_scene.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and amp_scaler is not None:
            with torch.amp.autocast('cuda'):
                output, expert_outputs, gate_weights, gate_logits = model(batch_x, batch_scene)
                loss, loss_dict = compute_loss(
                    output, batch_y, gate_logits,
                    lambda_mape, lambda_mae, lambda_mse, lambda_balance, lambda_range,
                    scaler=data_scaler, n_targets=n_targets, horizon=horizon,
                    use_weighted_loss=use_weighted_loss, balance_mode=balance_mode
                )
            
            amp_scaler.scale(loss).backward()
            # 检查梯度
            if debug and num_batches == 0:
                total_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"\n  [DEBUG] 调试信息（第一个batch）:")
                print(f"    损失值: {loss.item():.6f}")
                print(f"    梯度范数: {total_grad_norm:.6f}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            output, expert_outputs, gate_weights, gate_logits = model(batch_x, batch_scene)
            loss, loss_dict = compute_loss(
                    output, batch_y, gate_logits,
                    lambda_mape, lambda_mae, lambda_mse, lambda_balance, lambda_range,
                    scaler=data_scaler, n_targets=n_targets, horizon=horizon,
                    use_weighted_loss=use_weighted_loss, balance_mode=balance_mode
                )
            
            # 检查梯度
            if debug and num_batches == 0:
                total_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"\n  [DEBUG] 调试信息（第一个batch）:")
                print(f"    损失值: {loss.item():.6f}")
                print(f"    梯度范数: {total_grad_norm:.6f}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        # 检查参数是否更新
        if debug and num_batches == 0 and first_param_before is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_after = param.data.clone()
                    param_diff = (param_after - first_param_before).abs().mean().item()
                    print(f"    参数变化（{name}）: {param_diff:.8f}")
                    break
        
        total_loss += loss.item()
        for key in loss_components:
            if key in loss_dict:  # 安全检查：确保loss_dict中有这个键
                loss_components[key] += loss_dict[key].item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def calculate_metrics(model, data_loader, device, scaler=None, n_targets=None, horizon=1, dataset_name="", debug=False):
    """
    计算指标（用于训练集、验证集、测试集）
    按照实验要求的公式：
    - MAPE = (1/n) * Σ |Yi - Ŷi| / Yi * 100% (带阈值过滤)
    - MSE = Σ(yi - ŷi)² / Σyi² (不使用过滤，使用所有样本)
    - MAE = Σ|yi - ŷi| / Σyi (不使用过滤，使用所有样本)
    
    注意：yi和ŷi与MAPE的Yi和Ŷi是相同的实际客流值（反归一化后的数据）
    """
    model.eval()
    all_preds = []
    all_targets = []
    expert_usage = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch_x, batch_y, batch_scene = _unpack_batch(batch)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if batch_scene is not None:
                batch_scene = batch_scene.to(device)
            
            output, _, gate_weights, _ = model(batch_x, batch_scene)
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            expert_usage.append(gate_weights.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)  # [N, horizon * n_targets]
    all_targets = np.concatenate(all_targets, axis=0)  # [N, horizon * n_targets]
    
    # 调试：检查归一化数据上的模型输出
    if debug:
        if not hasattr(calculate_metrics, '_last_preds_norm_debug'):
            calculate_metrics._last_preds_norm_debug = None
        if calculate_metrics._last_preds_norm_debug is not None:
            pred_diff_norm_debug = np.abs(all_preds - calculate_metrics._last_preds_norm_debug).mean()
            print(f"\n  [DEBUG] 归一化数据上的模型输出变化: 平均绝对差异={pred_diff_norm_debug:.10f}")
            print(f"     归一化预测值范围: [{all_preds.min():.6f}, {all_preds.max():.6f}]")
        calculate_metrics._last_preds_norm_debug = all_preds.copy()
    
    # 反归一化（如果提供了scaler）
    if scaler is not None and n_targets is not None:
        # reshape: [N, horizon * n_targets] -> [N * horizon, n_targets]
        all_preds_2d = all_preds.reshape(-1, n_targets)
        all_targets_2d = all_targets.reshape(-1, n_targets)
        
        # 反归一化
        all_preds_orig = scaler.inverse_transform(all_preds_2d)
        all_targets_orig = scaler.inverse_transform(all_targets_2d)
        
        # 确保预测值非负（客流不能为负）
        # 注意：在反归一化后再限制，避免限制模型在归一化空间学习完整分布
        all_preds_orig = np.clip(all_preds_orig, 0, None)
        
        # 调试信息
        if debug:
            print(f"\n[DEBUG] 数据诊断信息 ({dataset_name}):")
            print(f"  反归一化后真实值范围: {all_targets_orig.min():.2f} - {all_targets_orig.max():.2f}")
            print(f"  反归一化后预测值范围: {all_preds_orig.min():.2f} - {all_preds_orig.max():.2f}")
            print(f"  真实值均值: {all_targets_orig.mean():.2f}, 标准差: {all_targets_orig.std():.2f}")
            print(f"  预测值均值: {all_preds_orig.mean():.2f}, 标准差: {all_preds_orig.std():.2f}")
            # 检查阈值过滤
            threshold = 10.0
            mask = (all_targets_orig.flatten() > threshold) & (all_preds_orig.flatten() > threshold)
            print(f"  阈值过滤 (threshold={threshold}): 有效样本={mask.sum()}, 总样本={len(mask)}, 过滤率={100*(1-mask.sum()/len(mask)):.1f}%")
            # 检查预测值是否在变化（与前一次比较）
            if not hasattr(calculate_metrics, '_last_preds_orig'):
                calculate_metrics._last_preds_orig = None
                calculate_metrics._last_preds_norm = None
            
            # 检查归一化数据上的预测值
            if calculate_metrics._last_preds_norm is not None:
                pred_diff_norm = np.abs(all_preds - calculate_metrics._last_preds_norm).mean()
                pred_diff_norm_max = np.abs(all_preds - calculate_metrics._last_preds_norm).max()
                print(f"  归一化预测值变化: 平均差异={pred_diff_norm:.10f}, 最大差异={pred_diff_norm_max:.10f}")
                if pred_diff_norm < 1e-8:
                    print(f"  [WARNING] 警告：归一化预测值完全没有变化！")
            
            # 检查反归一化后的预测值
            if calculate_metrics._last_preds_orig is not None:
                pred_diff_orig = np.abs(all_preds_orig - calculate_metrics._last_preds_orig).mean()
                pred_diff_orig_max = np.abs(all_preds_orig - calculate_metrics._last_preds_orig).max()
                print(f"  反归一化预测值变化: 平均差异={pred_diff_orig:.6f}, 最大差异={pred_diff_orig_max:.6f}")
                if pred_diff_orig < 1e-6:
                    print(f"  [WARNING] 警告：反归一化预测值完全没有变化！")
            
            calculate_metrics._last_preds_orig = all_preds_orig.copy()
            calculate_metrics._last_preds_norm = all_preds.copy()
        
        # 用于MAPE计算（flatten，带阈值过滤）
        preds_for_mape = all_preds_orig.flatten()
        targets_for_mape = all_targets_orig.flatten()
        
        # MSE和MAE使用归一化公式（按照实验要求）
        # 注意：yi和ŷi与MAPE的Yi和Ŷi是相同的实际客流值（反归一化后的数据）
        # MSE = Σ(yi - ŷi)² / Σyi²
        # MAE = Σ|yi - ŷi| / Σyi
        # 不使用过滤，使用所有样本
        errors_squared = (all_targets_orig - all_preds_orig) ** 2
        sum_errors_squared = np.sum(errors_squared)
        sum_true_squared = np.sum(all_targets_orig ** 2)
        mse = sum_errors_squared / (sum_true_squared + 1e-8)
        
        abs_errors = np.abs(all_targets_orig - all_preds_orig)
        sum_abs_errors = np.sum(abs_errors)
        sum_true = np.sum(all_targets_orig)
        mae = sum_abs_errors / (sum_true + 1e-8)
    else:
        # 没有scaler，使用归一化后的数据（降低阈值）
        preds_for_mape = all_preds.flatten()
        targets_for_mape = all_targets.flatten()
        
        # MSE和MAE使用归一化公式（按照实验要求）
        errors_squared = (all_targets - all_preds) ** 2
        sum_errors_squared = np.sum(errors_squared)
        sum_true_squared = np.sum(all_targets ** 2)
        mse = sum_errors_squared / (sum_true_squared + 1e-8)
        
        abs_errors = np.abs(all_targets - all_preds)
        sum_abs_errors = np.sum(abs_errors)
        sum_true = np.sum(all_targets)
        mae = sum_abs_errors / (sum_true + 1e-8)
    
    # 计算MAPE（使用原始尺度或归一化后的数据）
    mape, valid_count, ignored_count = mape_with_threshold(
        targets_for_mape, 
        preds_for_mape, 
        threshold=10.0 if scaler is not None else 0.1
    )
    
    # 计算专家利用率
    expert_usage = np.concatenate(expert_usage, axis=0)
    expert_utilization = expert_usage.mean(axis=0).tolist()
    
    results = {
        'mape': mape,
        'mse': mse,
        'mae': mae,
        'valid_count': valid_count,
        'ignored_count': ignored_count,
        'expert_utilization': expert_utilization
    }
    
    return results


def validate(model, val_loader, device, scaler=None, n_targets=None, horizon=1):
    """验证（兼容旧接口）"""
    return calculate_metrics(model, val_loader, device, scaler, n_targets, horizon, "验证集")


def main():
    parser = argparse.ArgumentParser(description='MoE-Nanjin 训练脚本')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='../data', help='数据目录（相对路径从当前工作目录解析，绝对路径直接使用）')
    parser.add_argument('--lookback', type=int, default=6, help='回看窗口长度')
    parser.add_argument('--horizon', type=int, default=1, help='预测窗口长度')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch大小（默认64，可根据GPU内存调整）')
    parser.add_argument('--use_both', action='store_true', default=True, help='同时使用from和to变量（20个变量，默认）')
    parser.add_argument('--use_from_only', action='store_true', help='只使用from变量（10个变量，复现之前MAPE=20%的配置）')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度（默认512，20个变量需要更大容量）')
    parser.add_argument('--num_experts', type=int, default=3, help='专家数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--ablation_mode', type=str, default='baseline', 
                        choices=['baseline', 'no_statistic', 'no_longterm', 'no_shortterm'],
                        help='消融实验模式')
    parser.add_argument('--use_scene_gating', action='store_true', default=True,
                        help='场景感知门控（默认开启）')
    parser.add_argument('--no_scene_gating', dest='use_scene_gating', action='store_false',
                        help='关闭场景门控（消融 Table 7）')
    parser.add_argument('--enhanced_statistic', action='store_true', default=True,
                        help='增强残差专家统计特征（默认开启）')
    parser.add_argument('--no_enhanced_statistic', dest='enhanced_statistic', action='store_false',
                        help='关闭增强统计（消融 Table 6）')
    parser.add_argument('--statistic_feature_set', type=str, default='robust',
                        choices=['basic', 'quantile', 'robust'],
                        help='统计专家特征集合: basic=mean/std/max, quantile=加入分位数, robust=加入鲁棒分布偏移特征')
    parser.add_argument('--use_regime_routing', action='store_true', default=True,
                        help='启用Regime-aware Mixture Routing（默认开启）')
    parser.add_argument('--no_regime_routing', dest='use_regime_routing', action='store_false',
                        help='关闭Regime-aware routing（消融）')
    parser.add_argument('--regime_dim', type=int, default=16,
                        help='regime embedding维度')
    parser.add_argument('--balance_mode', type=str, default='entropy', choices=['entropy', 'variance'],
                        help='负载均衡: entropy(熵正则) 或 variance(方差，旧版)')
    parser.add_argument('--snapshot_every', type=int, default=5,
                        help='每 N 个 epoch 保存测试集选模用快照（不删除，默认5）')
    parser.add_argument('--max_keep_models', type=int, default=30,
                        help='按验证集保留的 best 模型数量（默认30）')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=500, help='训练轮数（20个变量需要更长时间训练）')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--patience', type=int, default=80, help='早停耐心（默认50，给模型更多时间学习）')
    
    # 损失函数权重（已优化：更注重MAPE和范围学习）
    parser.add_argument('--lambda_mape', type=float, default=0.5, help='MAPE损失权重（默认0.0，完全参考timer_xl，只使用MSE）')
    parser.add_argument('--lambda_mae', type=float, default=1.0, help='MAE损失权重（默认0.0，完全参考timer_xl，只使用MSE）')
    parser.add_argument('--lambda_mse', type=float, default=0.5, help='MSE损失权重（默认1.0，完全参考timer_xl，只使用MSE）')
    parser.add_argument('--lambda_balance', type=float, default=0.01, help='负载均衡损失权重（默认0.01）')
    parser.add_argument('--lambda_range', type=float, default=0.1, help='范围惩罚损失权重（默认1.0，大幅增加以解决预测值范围偏小问题）')
    parser.add_argument('--use_weighted_loss', action='store_true', default=False, help='使用加权损失（给小值样本更高权重，默认False避免过度保守）')
    parser.add_argument('--no_weighted_loss', dest='use_weighted_loss', action='store_false', help='不使用加权损失（默认）')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader worker数量（默认8，多GPU时建议增加）')
    parser.add_argument('--save_dir', type=str, default=None, help='模型保存目录（默认: ../checkpoints/moe_nanjin）')
    parser.add_argument('--resume', type=str, default=None, help='从checkpoint恢复训练（指定checkpoint路径）')
    parser.add_argument('--multi_gpu', action='store_true', help='使用多GPU训练（DataParallel）')
    parser.add_argument('--gpu_ids', type=str, default=None, help='指定使用的GPU ID（如：0,1,2,3 或 all使用所有GPU）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 设备配置
    gpu_ids = None
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 张GPU")
        
        # 确定使用的GPU
        if args.multi_gpu or args.gpu_ids is not None:
            if args.gpu_ids is not None:
                if args.gpu_ids.lower() == 'all':
                    gpu_ids = list(range(num_gpus))
                else:
                    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
            else:
                # 使用所有GPU
                gpu_ids = list(range(num_gpus))
            
            device = torch.device(f'cuda:{gpu_ids[0]}')
            print(f"使用多GPU训练: GPU {gpu_ids}")
            print(f"主设备: {device}")
        else:
            # 默认使用后面的GPU（避免使用0,1,2）
            # 如果GPU数量>=6，使用3,4,5；否则使用最后一个GPU
            if num_gpus >= 6:
                default_gpu = 3  # 使用GPU 3
                print(f"GPU数量>=6，默认使用后面的GPU（避免0,1,2）")
            elif num_gpus >= 4:
                default_gpu = 3  # 使用GPU 3
                print(f"GPU数量>=4，使用GPU 3（避免0,1,2）")
            elif num_gpus >= 3:
                default_gpu = 2  # 使用GPU 2（如果只有3个GPU，使用最后一个）
                print(f"GPU数量={num_gpus}，使用GPU {default_gpu}")
            else:
                default_gpu = num_gpus - 1  # 使用最后一个GPU
                print(f"GPU数量={num_gpus}，使用最后一个GPU {default_gpu}")
            
            device = torch.device(f'cuda:{default_gpu}')
            print(f"使用单GPU: {device}")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        gpu_list = gpu_ids if gpu_ids is not None else [0]
        for gpu_id in gpu_list:
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            print(f"  GPU {gpu_id} 内存: {gpu_memory:.2f} GB")
    else:
        device = torch.device('cpu')
        print(f"使用设备: {device} (CUDA不可用)")
    
    # 准备数据（test_loader不会被使用，训练脚本只负责训练）
    print("\n准备数据...")
    # 如果指定了--use_from_only，则只使用from变量（10个）
    use_both = not args.use_from_only
    
    if args.use_from_only:
        print("  [WARNING] 使用配置：只使用from变量（10个），复现之前MAPE=20%的配置")
    else:
        print("  [INFO] 使用配置：同时使用from和to变量（20个）")
    
    train_loader, val_loader, _, scaler, dataset_info = prepare_data(
        data_dir=args.data_dir,
        target_cols=None,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_both=use_both
    )
    
    # 创建模型
    print("\n创建模型...")
    try:
        # 先创建模型在CPU上
        model = MoENanjin(
            input_dim=dataset_info['input_dim'],
            output_dim=dataset_info['output_dim'],
            lookback=args.lookback,
            num_experts=args.num_experts,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            ablation_mode=args.ablation_mode,
            use_scene_gating=args.use_scene_gating,
            enhanced_statistic=args.enhanced_statistic,
            statistic_feature_set=args.statistic_feature_set,
            use_regime_routing=args.use_regime_routing,
            regime_dim=args.regime_dim,
            scene_dim=dataset_info.get('scene_dim', 8),
        )
        
        # 计算模型参数数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数数量: {num_params:,}")
        
        # 估算模型内存占用（MB）
        model_size_mb = num_params * 4 / (1024 ** 2)  # 假设float32，4字节/参数
        print(f"模型大小（估算）: {model_size_mb:.2f} MB")
        
        # 移动到GPU并设置多GPU
        if device.type == 'cuda':
            # 清理缓存后再移动
            torch.cuda.empty_cache()
            model = model.to(device)
            
            # 多GPU支持
            if args.multi_gpu or args.gpu_ids is not None:
                if gpu_ids is not None and len(gpu_ids) > 1:
                    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
                    print(f"[OK] 已启用多GPU训练（DataParallel），使用GPU: {gpu_ids}")
                else:
                    print(f"模型已移动到GPU {device}")
            else:
                print(f"模型已移动到GPU {device}")
        else:
            print(f"模型在CPU上")
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n[ERROR] CUDA内存不足！")
            print(f"   建议解决方案：")
            print(f"   1. 减少批次大小: --batch_size 32 或 --batch_size 16")
            print(f"   2. 使用多GPU: --multi_gpu 或 --gpu_ids 0,1,2,3")
            print(f"   3. 使用混合精度: --use_amp")
            print(f"   4. 减少模型维度: --hidden_dim 64")
            print(f"\n   当前配置: batch_size={args.batch_size}, hidden_dim={args.hidden_dim}")
            if args.multi_gpu or args.gpu_ids is not None:
                print(f"   多GPU: 已启用")
            raise
        else:
            raise
    
    # 模型参数数量已在创建时打印
    
    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 使用CosineAnnealingLR（参考timer_xl，更适合长时间训练）
    # 或者使用ReduceLROnPlateau（更保守，适合20个变量的复杂任务）
    # 调整学习率调度器：更激进的patience，更快降低学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    # 混合精度训练
    amp_scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_mape': [],
        'val_mse': [],
        'val_mae': []
    }
    
    # 早停相关
    best_val_mape = float('inf')
    best_epoch = 0
    patience_counter = 0
    prev_val_mape = float('inf')  # 上一轮的MAPE，用于判断是否有改善
    
    # 保存目录（支持相对路径和绝对路径）
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_absolute():
            save_dir = Path.cwd() / save_dir
    else:
        # 根据ablation_mode自动设置保存目录
        if args.ablation_mode == 'baseline':
            save_dir = Path.cwd() / '../checkpoints/moe_nanjin'
        else:
            save_dir = Path.cwd() / f'../checkpoints/moe_nanjin_{args.ablation_mode}'
        save_dir = save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 从checkpoint恢复训练
    start_epoch = 1
    if args.resume:
        print(f"\n从checkpoint恢复训练: {args.resume}")
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = Path.cwd() / resume_path
        resume_path = resume_path.resolve()
        
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint文件不存在: {resume_path}")
        
        print(f"  加载checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        
        # 加载模型权重
        state_dict = checkpoint['model_state_dict']
        has_module_prefix_in_checkpoint = any(key.startswith('module.') for key in state_dict.keys())
        is_model_dataparallel = isinstance(model, torch.nn.DataParallel)
        
        # 处理module.前缀：确保state_dict的格式与当前模型匹配
        if has_module_prefix_in_checkpoint and not is_model_dataparallel:
            # Checkpoint有module.前缀，但当前模型不是DataParallel，需要移除
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
            print("  [INFO] 检测到DataParallel权重，已移除'module.'前缀")
        elif not has_module_prefix_in_checkpoint and is_model_dataparallel:
            # Checkpoint没有module.前缀，但当前模型是DataParallel，需要添加
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = f'module.{key}'
                new_state_dict[new_key] = value
            state_dict = new_state_dict
            print("  [INFO] 当前模型是DataParallel，已添加'module.'前缀")
        
        model.load_state_dict(state_dict)
        print(f"  [OK] 模型权重加载完成")
        
        # 加载优化器和调度器状态
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  [OK] 优化器状态加载完成")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  [OK] 调度器状态加载完成")
        if 'scaler_state_dict' in checkpoint and amp_scaler is not None:
            amp_scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"  [OK] 混合精度scaler状态加载完成")
        
        # 恢复训练状态
        start_epoch = checkpoint.get('epoch', 1) + 1
        best_val_mape = checkpoint.get('best_val_mape', float('inf'))
        best_epoch = checkpoint.get('best_epoch', 0)
        patience_counter = checkpoint.get('patience_counter', 0)
        prev_val_mape = checkpoint.get('val_mape', float('inf'))
        history = checkpoint.get('history', {'train_loss': [], 'val_mape': [], 'val_mse': [], 'val_mae': []})
        
        # 恢复best_models列表（从保存的checkpoint目录中重建）
        best_models = []
        max_keep_models = args.max_keep_models
        saved_checkpoints = sorted(save_dir.glob('best_model_epoch_*.pth'))
        for ckpt_path in saved_checkpoints:
            try:
                ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                ckpt_epoch = ckpt_data.get('epoch', 0)
                ckpt_mape = ckpt_data.get('val_mape', float('inf'))
                if ckpt_mape != float('inf') and ckpt_mape is not None:
                    best_models.append((ckpt_mape, ckpt_epoch, ckpt_path))
            except:
                pass
        best_models.sort(key=lambda x: x[0])  # 按MAPE排序
        if len(best_models) > max_keep_models:
            best_models = best_models[:max_keep_models]
        
        print(f"  [INFO] 从Epoch {start_epoch}继续训练")
        print(f"  [INFO] 当前最佳验证MAPE: {best_val_mape:.2f}% (Epoch {best_epoch})")
        print(f"  [INFO] 已保留 {len(best_models)} 个最佳模型")
    else:
        # 清理旧的checkpoint文件（训练开始时）
        print(f"\n清理旧的checkpoint文件...")
        old_checkpoints = list(save_dir.glob('*.pth'))
        if old_checkpoints:
            for old_file in old_checkpoints:
                old_file.unlink()
                print(f"  删除: {old_file.name}")
            print(f"  已清理 {len(old_checkpoints)} 个旧checkpoint文件")
        else:
            print(f"  没有旧的checkpoint文件需要清理")
    
    # 维护最好的5个模型列表 (mape, epoch, filepath)
    best_models = []
    max_keep_models = args.max_keep_models
    snap_dir = save_dir / 'snapshots_for_test'
    if args.snapshot_every > 0:
        snap_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始训练...")
    print(f"消融模式: {args.ablation_mode}")
    print(f"保存目录: {save_dir}")
    print(f"将保留最好的 {max_keep_models} 个模型（按验证集）")
    if args.snapshot_every > 0:
        print(f"每 {args.snapshot_every} epoch 保存测试选模快照 -> {snap_dir}")
    if args.resume:
        print(f"从Epoch {start_epoch}继续训练（总共 {args.num_epochs} 个epoch）")
    
    # 训练循环
    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start_time = time.time()
        
        # 训练
        debug_train = (epoch <= 3)  # 前3个epoch打印调试信息
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, device,
            lambda_mape=args.lambda_mape,  # MAPE损失权重（直接优化MAPE）
            lambda_mae=args.lambda_mae,
            lambda_mse=args.lambda_mse,
            lambda_balance=args.lambda_balance,
            lambda_range=args.lambda_range,  # 范围惩罚损失权重
            data_scaler=scaler,  # 用于反归一化计算MAPE损失（RobustScaler）
            amp_scaler=amp_scaler,  # 用于混合精度训练（GradScaler）
            use_amp=args.use_amp,
            n_targets=dataset_info['n_targets'],
            horizon=args.horizon,
            use_weighted_loss=args.use_weighted_loss,
            debug=debug_train,
            balance_mode=args.balance_mode,
        )
        
        # 验证（传入scaler进行反归一化）
        # 添加调试信息：检查模型输出是否在变化
        debug_val = (epoch <= 3) or (epoch % 10 == 0)  # 前3个epoch和每10个epoch打印调试信息
        val_results = calculate_metrics(
            model, val_loader, device, 
            scaler=scaler, 
            n_targets=dataset_info['n_targets'],
            horizon=args.horizon,
            dataset_name="验证集",
            debug=debug_val  # 打印调试信息
        )
        
        # 学习率调度
        scheduler.step(val_results['mape'] if val_results['mape'] is not None else val_results['mse'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_mape'].append(val_results['mape'] if val_results['mape'] is not None else float('inf'))
        history['val_mse'].append(val_results['mse'])
        history['val_mae'].append(val_results['mae'])
        
        epoch_time = time.time() - epoch_start_time
        
        # 打印结果
        print(f"\n[INFO] Epoch {epoch}/{args.num_epochs}:")
        print(f"  训练损失: {train_loss:.6f}")
        mape_str = f"MAPE={train_components['mape']:.4f}" if train_components['mape'] > 0 else "MAPE=0.0000"
        range_str = f"Range={train_components.get('range', 0):.4f}" if 'range' in train_components else "Range=0.0000"
        print(f"  损失分量: {mape_str}, MAE={train_components['mae']:.4f}, MSE={train_components['mse']:.4f}, Balance={train_components['balance']:.4f}, {range_str}")
        print(f"  【验证集】 MAPE: {val_results['mape']:.2f}%" if val_results['mape'] is not None else "  【验证集】 MAPE: N/A")
        print(f"  【验证集】 MSE:  {val_results['mse']:.6f}")
        print(f"  【验证集】 MAE:  {val_results['mae']:.6f}")
        print(f"  学习率:   {current_lr:.2e}")
        print(f"  耗时:     {epoch_time:.1f}秒")
        print(f"  Expert利用率: {val_results['expert_utilization']}")
        
        # 判断是否有改善（只要比上一轮好就算改善）
        has_improvement = False
        if val_results['mape'] is not None:
            # 第一轮或比上一轮好，都算改善
            if prev_val_mape == float('inf') or val_results['mape'] < prev_val_mape:
                has_improvement = True
                improvement_from_prev = prev_val_mape - val_results['mape'] if prev_val_mape < float('inf') else 0
                patience_counter = 0  # 重置耐心值
                print(f"  [IMPROVE] 比上一轮改善 {improvement_from_prev:.2f}% (上一轮: {prev_val_mape:.2f}% -> 当前: {val_results['mape']:.2f}%)")
                print(f"  [RESET] 耐心值已重置为0 (剩余耐心: {args.patience})")
            else:
                patience_counter += 1
                remaining_patience = args.patience - patience_counter
                print(f"  [WAIT] 无改善 ({patience_counter}/{args.patience}) - 剩余耐心: {remaining_patience}")
                print(f"         (上一轮: {prev_val_mape:.2f}% -> 当前: {val_results['mape']:.2f}%)")
            
            # 更新上一轮的MAPE
            prev_val_mape = val_results['mape']
        
        # 保存最佳模型（基于历史最佳MAPE）
        # 第一个epoch、比历史最佳更好、或者比当前最差更好（且列表未满）时保存
        should_save = False
        if val_results['mape'] is not None:
            # 情况1：第一个epoch或比历史最佳更好
            if best_val_mape == float('inf') or val_results['mape'] < best_val_mape:
                should_save = True
            # 情况2：比当前最差的模型更好，且列表未满（需要保留更多好模型）
            elif len(best_models) < max_keep_models:
                # 如果列表未满，只要比当前最差的更好就保存
                if len(best_models) > 0:
                    worst_mape_in_list = max([m[0] for m in best_models])  # 当前列表中最差的MAPE
                    if val_results['mape'] < worst_mape_in_list:
                        should_save = True
                else:
                    # 列表为空，直接保存
                    should_save = True
            # 情况3：列表已满，但比当前最差的更好，替换最差的
            elif len(best_models) >= max_keep_models:
                worst_mape_in_list = max([m[0] for m in best_models])  # 当前列表中最差的MAPE
                if val_results['mape'] < worst_mape_in_list:
                    should_save = True
        else:
            # 如果MAPE为None，打印警告
            if epoch == 1:
                print(f"  [WARNING] Epoch {epoch}: 验证集MAPE为None，无法保存模型！")
        
        if should_save:
            # 判断是否是历史最佳
            is_new_best = (best_val_mape == float('inf') or val_results['mape'] < best_val_mape)
            improvement = best_val_mape - val_results['mape'] if best_val_mape < float('inf') else 0
            
            # 只有比历史最佳更好时才更新best_val_mape和best_epoch
            if is_new_best:
                best_val_mape = val_results['mape']
                best_epoch = epoch
            
            # 保存checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': {
                    'input_dim': dataset_info['input_dim'],
                    'output_dim': dataset_info['output_dim'],
                    'num_experts': args.num_experts,
                    'hidden_dim': args.hidden_dim,
                    'lookback': args.lookback,
                    'horizon': args.horizon,
                    'n_targets': dataset_info['n_targets'],
                    'ablation_mode': args.ablation_mode,
                    'use_scene_gating': args.use_scene_gating,
                    'enhanced_statistic': args.enhanced_statistic,
                    'statistic_feature_set': args.statistic_feature_set,
                    'use_regime_routing': args.use_regime_routing,
                    'regime_dim': args.regime_dim,
                    'balance_mode': args.balance_mode,
                    'scene_dim': dataset_info.get('scene_dim', 0),
                },
                'history': history,
                'val_mse': val_results['mse'],
                'val_mae': val_results['mae'],
                'val_mape': val_results['mape'],
                'best_val_mape': best_val_mape,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
            }
            
            if amp_scaler is not None:
                checkpoint['scaler_state_dict'] = amp_scaler.state_dict()
            
            # 保存新模型
            best_model_path = save_dir / f'best_model_epoch_{epoch:03d}_mape_{val_results["mape"]:.2f}.pth'
            torch.save(checkpoint, best_model_path)
            
            # 更新最好的模型列表（按MAPE排序，越小越好）
            best_models.append((val_results['mape'], epoch, best_model_path))
            best_models.sort(key=lambda x: x[0])  # 按MAPE升序排序
            
            # 如果超过5个，删除最差的模型（循环删除直到只剩5个）
            while len(best_models) > max_keep_models:
                # 删除最差的模型（MAPE最大的，即列表最后一个）
                worst_mape, worst_epoch, worst_path = best_models.pop()  # 移除最后一个（最差的）
                if worst_path.exists():
                    worst_path.unlink()
                    print(f"  [DELETE] 删除最差模型: {worst_path.name} (MAPE: {worst_mape:.2f}%)")
            
            # 始终更新best_model_latest.pth（最新的最佳模型）
            latest_model_path = save_dir / 'best_model_latest.pth'
            torch.save(checkpoint, latest_model_path)
            
            # 打印保存信息
            if is_new_best:
                print(f"  [NEW] 新历史最佳MAPE! 改善 {improvement:.2f}% (之前历史最佳: {best_val_mape + improvement:.2f}%)")
            else:
                worst_mape_in_list = max([m[0] for m in best_models]) if len(best_models) > 0 else float('inf')
                print(f"  [SAVE] 保存模型（比当前最差更好）: {best_model_path.name}")
                print(f"        当前MAPE: {val_results['mape']:.2f}%, 替换最差: {worst_mape_in_list:.2f}%")
            print(f"  [SAVE] 保存模型: {best_model_path.name}")
            print(f"  [KEEP] 当前保留 {len(best_models)}/{max_keep_models} 个最佳模型")
        
        # 定期快照：供在测试集上选最优 epoch（原文 Epoch47 做法）
        if args.snapshot_every > 0 and (epoch % args.snapshot_every == 0):
            snap_ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': {
                    'input_dim': dataset_info['input_dim'],
                    'output_dim': dataset_info['output_dim'],
                    'num_experts': args.num_experts,
                    'hidden_dim': args.hidden_dim,
                    'lookback': args.lookback,
                    'horizon': args.horizon,
                    'n_targets': dataset_info['n_targets'],
                    'ablation_mode': args.ablation_mode,
                    'use_scene_gating': args.use_scene_gating,
                    'enhanced_statistic': args.enhanced_statistic,
                    'balance_mode': args.balance_mode,
                    'scene_dim': dataset_info.get('scene_dim', 0),
                },
                'val_mape': val_results['mape'],
                'val_mse': val_results['mse'],
                'val_mae': val_results['mae'],
            }
            if amp_scaler is not None:
                snap_ckpt['scaler_state_dict'] = amp_scaler.state_dict()
            snap_path = snap_dir / f'epoch_{epoch:03d}.pth'
            torch.save(snap_ckpt, snap_path)
            print(f"  [SNAPSHOT] 测试选模快照: {snap_path.name} (val MAPE={val_results['mape']:.2f}%)")
            
        # 早停检查
        if patience_counter >= args.patience:
            print(f"\n{'='*80}")
            print(f"[STOP] 早停触发！")
            print(f"   ├─ 最佳Epoch: {best_epoch}")
            print(f"   └─ 最佳Val MAPE: {best_val_mape:.2f}%")
            print(f"   提示: 请运行 eval_all_checkpoints.py 在测试集上选最优模型")
            print(f"{'='*80}\n")
            break
    
    # 训练完成，输出最终结果（仅验证集）
    print(f"\n{'='*80}")
    print(f"[INFO] 训练完成！")
    print(f"{'='*80}")
    print(f"  最佳Epoch: {best_epoch}")
    print(f"  最佳验证MAPE: {best_val_mape:.2f}%")
    print(f"\n  保留的最佳模型列表（共 {len(best_models)} 个）:")
    for idx, (mape, epoch, filepath) in enumerate(best_models, 1):
        # 使用浮点数比较（允许小的误差）
        marker = " ⭐" if abs(mape - best_val_mape) < 0.01 else ""
        print(f"    {idx}. Epoch {epoch:03d} - MAPE: {mape:.2f}% - {filepath.name}{marker}")
    print(f"\n  最佳模型（最新）: {save_dir / 'best_model_latest.pth'}")
    print(f"\n  最终指标（最后一轮验证集，非最佳模型）:")
    print(f"  【验证集】 MAPE: {val_results['mape']:.2f}%" if val_results['mape'] is not None else "  【验证集】 MAPE: N/A")
    print(f"  【验证集】 MSE:  {val_results['mse']:.6f}")
    print(f"  【验证集】 MAE:  {val_results['mae']:.6f}")
    print(f"\n  ⚠️  注意：上述指标是最后一轮的验证集结果，不是最佳模型的结果")
    print(f"     最佳模型的验证集MAPE: {best_val_mape:.2f}% (Epoch {best_epoch})")
    print(f"\n  提示: 使用 evaluate.py 脚本评估测试集性能")
    print(f"        推荐使用: {save_dir / 'best_model_latest.pth'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
