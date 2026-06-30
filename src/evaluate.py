"""
MoE-Nanjin 评估脚本
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from model import MoENanjin
from data_loader import prepare_data
from utils import mape_with_threshold, mse_loss, mae_loss


def evaluate(model, test_loader, device, scaler=None):
    """评估模型"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_scene = batch
                batch_scene = batch_scene.to(device)
            else:
                batch_x, batch_y = batch
                batch_scene = None
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output, _, _, _ = model(batch_x, batch_scene)
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 如果提供了scaler，反标准化到原始尺度（用于MAPE计算）
    if scaler is not None:
        # 输出是flatten的 [N, horizon * n_targets]
        # 需要reshape为 [N * horizon, n_targets] 才能反标准化
        n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else all_targets.shape[1]
        
        # 如果输出是flatten的 [N, horizon * n_targets]，需要reshape
        if len(all_targets.shape) == 2 and all_targets.shape[1] > n_features:
            # 是flatten的，需要reshape
            horizon = all_targets.shape[1] // n_features
            all_targets_2d = all_targets.reshape(-1, n_features)
            all_preds_2d = all_preds.reshape(-1, n_features)
        else:
            # 已经是正确的shape [N, n_features]
            all_targets_2d = all_targets
            all_preds_2d = all_preds
        
        # 反标准化
        all_targets_orig = scaler.inverse_transform(all_targets_2d)
        all_preds_orig = scaler.inverse_transform(all_preds_2d)
        
        # 确保预测值非负（客流不能为负）
        # 注意：在反归一化后再限制，避免限制模型在归一化空间学习完整分布
        all_preds_orig = np.clip(all_preds_orig, 0, None)
        
        # 计算MAPE（使用原始尺度，flatten后计算，带阈值过滤）
        mape, valid_count, ignored_count = mape_with_threshold(
            all_targets_orig.flatten(), 
            all_preds_orig.flatten(), 
            threshold=10.0
        )
        
        # MSE和MAE使用归一化公式（按照实验要求，在原始尺度上计算）
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
        # 没有scaler，直接计算（标准化后的数据，降低阈值）
        mape, valid_count, ignored_count = mape_with_threshold(
            all_targets.flatten(), 
            all_preds.flatten(), 
            threshold=0.1
        )
        
        # MSE和MAE使用归一化公式（按照实验要求，在归一化尺度上计算）
        errors_squared = (all_targets - all_preds) ** 2
        sum_errors_squared = np.sum(errors_squared)
        sum_true_squared = np.sum(all_targets ** 2)
        mse = sum_errors_squared / (sum_true_squared + 1e-8)
        
        abs_errors = np.abs(all_targets - all_preds)
        sum_abs_errors = np.sum(abs_errors)
        sum_true = np.sum(all_targets)
        mae = sum_abs_errors / (sum_true + 1e-8)
    
    results = {
        'mape': mape,
        'mse': mse,
        'mae': mae,
        'valid_count': valid_count,
        'ignored_count': ignored_count
    }
    
    # 返回反标准化后的数据（如果可用）
    if scaler is not None:
        return results, all_preds_orig, all_targets_orig, all_preds, all_targets
    else:
        # 没有scaler时，使用原始数据作为"反标准化"数据
        return results, all_preds, all_targets, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description='MoE-Nanjin 评估脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--data_dir', type=str, default='../data', help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch大小')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker数量')
    
    args = parser.parse_args()
    
    # 设备（指定cuda:0）
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"使用设备: {device}")
    else:
        device = torch.device('cpu')
        print(f"使用设备: {device} (CUDA不可用)")
    
    # 加载checkpoint
    print(f"\n加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # 从checkpoint读取ablation_mode，如果没有则从路径推断
    ablation_mode = config.get('ablation_mode', None)
    if ablation_mode is None:
        # 从checkpoint路径推断ablation_mode
        checkpoint_path = Path(args.checkpoint)
        if 'no_statistic' in str(checkpoint_path):
            ablation_mode = 'no_statistic'
        elif 'no_longterm' in str(checkpoint_path):
            ablation_mode = 'no_longterm'
        elif 'no_shortterm' in str(checkpoint_path):
            ablation_mode = 'no_shortterm'
        else:
            ablation_mode = 'baseline'
        print(f"  [INFO] 从路径推断消融模式: {ablation_mode}")
    else:
        print(f"  [INFO] 从checkpoint读取消融模式: {ablation_mode}")
    
    # 创建模型
    print("\n创建模型...")
    model = MoENanjin(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        lookback=config['lookback'],
        num_experts=config['num_experts'],
        hidden_dim=config['hidden_dim'],
        dropout=0.1,
        ablation_mode=ablation_mode,
        use_scene_gating=config.get('use_scene_gating', False),
        enhanced_statistic=config.get('enhanced_statistic', True),
        statistic_feature_set=config.get('statistic_feature_set', 'robust'),
        use_regime_routing=config.get('use_regime_routing', False),
        regime_dim=config.get('regime_dim', 16),
        scene_dim=config.get('scene_dim', 0),
    ).to(device)
    
    # 加载权重，处理DataParallel的module前缀
    state_dict = checkpoint['model_state_dict']
    
    # 检查是否有module前缀（DataParallel保存的模型）
    has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
    
    if has_module_prefix:
        # 去掉module前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 去掉'module.'前缀
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
        print("  [INFO] 检测到DataParallel权重，已移除'module.'前缀")
    
    model.load_state_dict(state_dict)
    print("[OK] 模型加载完成")
    
    # 准备数据
    print("\n准备数据...")
    _, _, test_loader, scaler, dataset_info = prepare_data(
        data_dir=args.data_dir,
        target_cols=None,
        lookback=config['lookback'],
        horizon=config['horizon'],
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 评估（传入scaler用于反标准化计算MAPE）
    print("\n开始评估...")
    eval_result = evaluate(model, test_loader, device, scaler=scaler)
    
    if scaler is not None:
        results, preds_orig, targets_orig, preds, targets = eval_result
    else:
        results, preds, targets = eval_result
        preds_orig = preds
        targets_orig = targets
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"[INFO] 测试集评估结果:")
    print(f"{'='*80}")
    print(f"  MAPE: {results['mape']:.2f}%" if results['mape'] is not None else "  MAPE: N/A")
    print(f"  MSE:  {results['mse']:.6f}")
    print(f"  MAE:  {results['mae']:.6f}")
    print(f"  有效样本: {results['valid_count']}")
    print(f"  忽略样本: {results['ignored_count']}")
    print(f"{'='*80}\n")
    
    # 保存结果
    result_dir = Path('../results/moe_nanjin')
    if not result_dir.is_absolute():
        result_dir = Path.cwd() / result_dir
        result_dir = result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = result_dir / 'evaluation_results.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("测试集评估结果\n")
        f.write("="*80 + "\n")
        f.write(f"MAPE: {results['mape']:.2f}%\n" if results['mape'] is not None else "MAPE: N/A\n")
        f.write(f"MSE:  {results['mse']:.6f}\n")
        f.write(f"MAE:  {results['mae']:.6f}\n")
        f.write(f"有效样本: {results['valid_count']}\n")
        f.write(f"忽略样本: {results['ignored_count']}\n")
    
    print(f"[OK] 结果已保存到: {result_file}")
    
    # 生成真实值和预测值对比文件
    # 获取目标列名
    target_cols = dataset_info.get('target_cols', [f'变量{i+1}' for i in range(dataset_info['n_targets'])])
    n_targets = dataset_info['n_targets']
    horizon = config['horizon']
    
    # 如果输出是flatten的，需要reshape
    if len(preds_orig.shape) == 2 and preds_orig.shape[1] == n_targets * horizon:
        # reshape为 [N * horizon, n_targets]
        preds_reshaped = preds_orig.reshape(-1, n_targets)
        targets_reshaped = targets_orig.reshape(-1, n_targets)
    else:
        preds_reshaped = preds_orig
        targets_reshaped = targets_orig
    
    # 创建DataFrame，每个变量的真实值和预测值挨在一起
    data_dict = {}
    for i, col_name in enumerate(target_cols):
        data_dict[f'{col_name}_真实值'] = targets_reshaped[:, i]
        data_dict[f'{col_name}_预测值'] = preds_reshaped[:, i]
    
    comparison_df = pd.DataFrame(data_dict)
    
    # 保存为CSV文件
    comparison_file = result_dir / 'test_predictions_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    
    print(f"[OK] 真实值和预测值对比已保存到: {comparison_file}")
    print(f"   共 {len(comparison_df)} 行，{len(comparison_df.columns)} 列（{n_targets}个变量 × 2）")


if __name__ == '__main__':
    main()
