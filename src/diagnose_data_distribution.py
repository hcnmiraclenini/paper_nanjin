"""
诊断数据分布差异
检查训练集、验证集、测试集的数据分布是否一致
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader import load_and_merge_data
from sklearn.preprocessing import RobustScaler

def main():
    # 加载数据
    data_dir = Path('data')
    merged_df, target_cols = load_and_merge_data(data_dir, use_both=True)
    
    # 提取目标数据
    target_data = merged_df[target_cols].values  # [N, n_targets]
    
    # 标准化（使用RobustScaler，与训练时一致）
    scaler = RobustScaler()
    target_data_scaled = scaler.fit_transform(target_data)
    
    # 按时间顺序划分数据集（与训练时一致）
    total_len = len(target_data_scaled)
    train_size = int(total_len * 0.5)
    val_size = int(total_len * 0.25)
    test_size = total_len - train_size - val_size
    
    train_data = target_data_scaled[:train_size]
    val_data = target_data_scaled[train_size:train_size+val_size]
    test_data = target_data_scaled[train_size+val_size:]
    
    print("="*80)
    print("数据分布诊断")
    print("="*80)
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_data)} ({len(train_data)/total_len*100:.1f}%)")
    print(f"  验证集: {len(val_data)} ({len(val_data)/total_len*100:.1f}%)")
    print(f"  测试集: {len(test_data)} ({len(test_data)/total_len*100:.1f}%)")
    
    # 检查数据分布（归一化后的数据）
    print(f"\n归一化后数据分布统计:")
    print(f"  训练集均值: {train_data.mean():.6f}, 标准差: {train_data.std():.6f}")
    print(f"  验证集均值: {val_data.mean():.6f}, 标准差: {val_data.std():.6f}")
    print(f"  测试集均值: {test_data.mean():.6f}, 标准差: {test_data.std():.6f}")
    print(f"  验证集/训练集比例: {val_data.mean()/train_data.mean():.4f}")
    print(f"  测试集/训练集比例: {test_data.mean()/train_data.mean():.4f}")
    
    # 检查原始数据分布（反归一化）
    train_data_orig = scaler.inverse_transform(train_data)
    val_data_orig = scaler.inverse_transform(val_data)
    test_data_orig = scaler.inverse_transform(test_data)
    
    print(f"\n原始数据分布统计（反归一化后）:")
    print(f"  训练集均值: {train_data_orig.mean():.2f}, 标准差: {train_data_orig.std():.2f}")
    print(f"  验证集均值: {val_data_orig.mean():.2f}, 标准差: {val_data_orig.std():.2f}")
    print(f"  测试集均值: {test_data_orig.mean():.2f}, 标准差: {test_data_orig.std():.2f}")
    print(f"  验证集/训练集比例: {val_data_orig.mean()/train_data_orig.mean():.4f}")
    print(f"  测试集/训练集比例: {test_data_orig.mean()/train_data_orig.mean():.4f}")
    
    # 检查每个变量的分布
    print(f"\n各变量分布差异（前10个变量）:")
    for i, col in enumerate(target_cols[:10]):
        train_mean = train_data_orig[:, i].mean()
        val_mean = val_data_orig[:, i].mean()
        test_mean = test_data_orig[:, i].mean()
        val_ratio = val_mean / train_mean if train_mean > 0 else 0
        test_ratio = test_mean / train_mean if train_mean > 0 else 0
        print(f"  {col}:")
        print(f"    训练集均值: {train_mean:.2f}")
        print(f"    验证集均值: {val_mean:.2f} (比例: {val_ratio:.4f})")
        print(f"    测试集均值: {test_mean:.2f} (比例: {test_ratio:.4f})")
        if abs(val_ratio - 1.0) > 0.1 or abs(test_ratio - 1.0) > 0.1:
            print(f"    ⚠️  警告：该变量分布差异较大！")
    
    # 检查时间范围
    print(f"\n时间范围:")
    print(f"  训练集: {merged_df.iloc[0]['time']} 到 {merged_df.iloc[train_size-1]['time']}")
    print(f"  验证集: {merged_df.iloc[train_size]['time']} 到 {merged_df.iloc[train_size+val_size-1]['time']}")
    print(f"  测试集: {merged_df.iloc[train_size+val_size]['time']} 到 {merged_df.iloc[-1]['time']}")
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)
    print("\n建议:")
    print("1. 如果数据分布差异较大，可能需要重新划分数据集")
    print("2. 如果测试集数据更容易预测（均值更小或更稳定），MAPE差异是正常的")
    print("3. 如果验证集和测试集分布差异很大，建议使用交叉验证或重新划分")

if __name__ == '__main__':
    main()
