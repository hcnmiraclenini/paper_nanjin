"""
批量评估所有checkpoint的测试集MAPE
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from model import MoENanjin
from data_loader import prepare_data
from utils import mape_with_threshold


def evaluate_checkpoint(checkpoint_path, device, data_cache=None):
    """评估单个checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    
    # 检查文件是否有效
    if not checkpoint_path.exists():
        return None, f"文件不存在"
    if checkpoint_path.stat().st_size == 0:
        return None, f"文件为空"
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # 从checkpoint读取ablation_mode
        ablation_mode = config.get('ablation_mode', 'baseline')
        
        # 获取lookback
        lookback = config.get('lookback', 6)
        
        # 准备数据（使用缓存避免重复加载）
        cache_key = f"lookback_{lookback}"
        if data_cache is not None and cache_key in data_cache:
            test_loader, scaler, dataset_info = data_cache[cache_key]
        else:
            _, _, test_loader, scaler, dataset_info = prepare_data(
                data_dir='data',
                target_cols=None,
                lookback=lookback,
                horizon=config.get('horizon', 1),
                batch_size=32,
                num_workers=4
            )
            if data_cache is not None:
                data_cache[cache_key] = (test_loader, scaler, dataset_info)
        
        # 创建模型
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
            scene_dim=config.get('scene_dim', 0),
        ).to(device)
        
        # 加载权重
        state_dict = checkpoint['model_state_dict']
        has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
        if has_module_prefix:
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # 评估
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
        
        # 反标准化
        n_features = scaler.n_features_in_
        if len(all_targets.shape) == 2 and all_targets.shape[1] > n_features:
            all_targets_2d = all_targets.reshape(-1, n_features)
            all_preds_2d = all_preds.reshape(-1, n_features)
        else:
            all_targets_2d = all_targets
            all_preds_2d = all_preds
        
        all_targets_orig = scaler.inverse_transform(all_targets_2d)
        all_preds_orig = scaler.inverse_transform(all_preds_2d)
        all_preds_orig = np.clip(all_preds_orig, 0, None)
        
        # 计算MAPE
        mape, valid_count, ignored_count = mape_with_threshold(
            all_targets_orig.flatten(),
            all_preds_orig.flatten(),
            threshold=10.0
        )
        
        return {
            'mape': mape,
            'valid_count': valid_count,
            'ignored_count': ignored_count,
            'lookback': lookback
        }, None
        
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description='批量评估checkpoint')
    parser.add_argument('--data_dir', type=str, default='../data', help='数据目录')
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 定义要测试的checkpoint目录和文件
    checkpoint_configs = [
        # (目录名, 最佳checkpoint文件名)
        ('moe_nanjin', 'best_model_epoch_002_mape_39.44.pth'),
        ('moe_nanjin_lookback_6', 'best_model_epoch_009_mape_32.28.pth'),
        ('moe_nanjin_lookback_7', 'best_model_epoch_023_mape_31.20.pth'),
        ('moe_nanjin_lookback_10', 'best_model_epoch_006_mape_33.13.pth'),
        ('moe_nanjin_lookback_14', 'best_model_epoch_006_mape_31.77.pth'),
        ('moe_nanjin_lookback_30', 'best_model_epoch_097_mape_32.42.pth'),
    ]
    
    base_dir = Path('../checkpoints')
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    
    # 数据缓存
    data_cache = {}
    
    # 结果
    results = []
    
    print("\n" + "=" * 80)
    print("开始批量评估...")
    print("=" * 80)
    
    for dir_name, best_file in checkpoint_configs:
        checkpoint_dir = base_dir / dir_name
        checkpoint_path = checkpoint_dir / best_file
        
        print(f"\n[评估] {dir_name}/{best_file}")
        
        result, error = evaluate_checkpoint(checkpoint_path, device, data_cache)
        
        if result is not None:
            print(f"  ✓ 测试集MAPE: {result['mape']:.2f}% (lookback={result['lookback']})")
            results.append({
                'dir': dir_name,
                'file': best_file,
                'mape': result['mape'],
                'lookback': result['lookback'],
                'valid_count': result['valid_count']
            })
        else:
            print(f"  ✗ 错误: {error}")
            results.append({
                'dir': dir_name,
                'file': best_file,
                'mape': None,
                'lookback': None,
                'error': error
            })
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("评估结果汇总（按MAPE排序）")
    print("=" * 80)
    
    # 过滤有效结果并排序
    valid_results = [r for r in results if r['mape'] is not None]
    valid_results.sort(key=lambda x: x['mape'])
    
    print(f"\n{'排名':<4} {'目录':<30} {'Lookback':<10} {'测试集MAPE':<12}")
    print("-" * 60)
    
    for i, r in enumerate(valid_results, 1):
        marker = "🏆" if i == 1 else "  "
        print(f"{marker}{i:<3} {r['dir']:<30} {r['lookback']:<10} {r['mape']:.2f}%")
    
    if valid_results:
        best = valid_results[0]
        print("\n" + "=" * 80)
        print(f"🏆 最佳模型: {best['dir']}/{best['file']}")
        print(f"   Lookback: {best['lookback']}")
        print(f"   测试集MAPE: {best['mape']:.2f}%")
        print("=" * 80)


if __name__ == '__main__':
    main()
