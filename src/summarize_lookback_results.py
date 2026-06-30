#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总回看窗口（Lookback）消融实验结果
"""

import re
import os
from pathlib import Path
from collections import defaultdict

def extract_evaluation_results(file_path):
    """从评估结果文件中提取指标"""
    results = {}
    
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取MAPE
    mape_match = re.search(r'MAPE:\s*([\d.]+)%', content)
    if mape_match:
        results['MAPE'] = float(mape_match.group(1))
    
    # 提取MSE
    mse_match = re.search(r'MSE:\s*([\d.]+)', content)
    if mse_match:
        results['MSE'] = float(mse_match.group(1))
    
    # 提取MAE
    mae_match = re.search(r'MAE:\s*([\d.]+)', content)
    if mae_match:
        results['MAE'] = float(mae_match.group(1))
    
    # 提取有效样本数
    valid_match = re.search(r'有效样本:\s*(\d+)', content)
    if valid_match:
        results['有效样本'] = int(valid_match.group(1))
    
    # 提取忽略样本数
    ignored_match = re.search(r'忽略样本:\s*(\d+)', content)
    if ignored_match:
        results['忽略样本'] = int(ignored_match.group(1))
    
    return results if results else None

def main():
    results_dir = Path("../results/moe_nanjin_lookback_ablation")
    
    if not results_dir.exists():
        print(f"结果目录不存在: {results_dir}")
        return
    
    # 查找所有评估结果文件
    evaluation_files = sorted(results_dir.glob("lookback_*_evaluation.txt"))
    
    if not evaluation_files:
        print("未找到评估结果文件")
        return
    
    # 提取所有结果
    all_results = []
    for eval_file in evaluation_files:
        # 从文件名提取lookback值
        match = re.search(r'lookback_(\d+)_evaluation', eval_file.name)
        if match:
            lookback = int(match.group(1))
            results = extract_evaluation_results(eval_file)
            if results:
                results['Lookback'] = lookback
                all_results.append(results)
    
    if not all_results:
        print("未能提取到任何结果")
        return
    
    # 按Lookback排序
    all_results.sort(key=lambda x: x['Lookback'])
    
    # 生成汇总报告
    output_file = results_dir / "lookback_ablation_summary.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MoE-Nanjin 回看窗口（Lookback）消融实验结果汇总\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("实验配置：\n")
        f.write("-" * 80 + "\n")
        f.write("模型：基线模型（3个专家：ShortTerm + LongTerm + Statistic）\n")
        f.write("其他配置：与基线模型完全一致（horizon=1, hidden_dim=512, dropout=0.15等）\n")
        f.write("测试的Lookback值：6, 7, 10, 14, 30\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("实验结果\n")
        f.write("=" * 80 + "\n\n")
        
        # 表格头部
        f.write(f"{'Lookback':<10} {'MAPE (%)':<12} {'MSE':<12} {'MAE':<12} {'有效样本':<12} {'忽略样本':<12}\n")
        f.write("-" * 80 + "\n")
        
        # 找出最佳结果
        best_mape = float('inf')
        best_lookback = None
        
        for result in all_results:
            lookback = result['Lookback']
            mape = result.get('MAPE', 0)
            mse = result.get('MSE', 0)
            mae = result.get('MAE', 0)
            valid = result.get('有效样本', 0)
            ignored = result.get('忽略样本', 0)
            
            f.write(f"{lookback:<10} {mape:<12.2f} {mse:<12.6f} {mae:<12.6f} {valid:<12} {ignored:<12}\n")
            
            if mape < best_mape:
                best_mape = mape
                best_lookback = lookback
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("结果分析\n")
        f.write("=" * 80 + "\n\n")
        
        # 找出最佳结果
        f.write(f"最佳结果：\n")
        f.write(f"  - Lookback = {best_lookback}\n")
        f.write(f"  - MAPE = {best_mape:.2f}%\n")
        f.write("\n")
        
        # 性能排序
        sorted_results = sorted(all_results, key=lambda x: x.get('MAPE', float('inf')))
        f.write("性能排序（按MAPE从低到高）：\n")
        for i, result in enumerate(sorted_results, 1):
            lookback = result['Lookback']
            mape = result.get('MAPE', 0)
            f.write(f"  {i}. Lookback={lookback}: MAPE={mape:.2f}%\n")
        f.write("\n")
        
        # 与基线（lookback=6）对比
        baseline_result = next((r for r in all_results if r['Lookback'] == 6), None)
        if baseline_result:
            baseline_mape = baseline_result.get('MAPE', 0)
            f.write(f"与基线（Lookback=6, MAPE={baseline_mape:.2f}%）对比：\n")
            for result in all_results:
                lookback = result['Lookback']
                mape = result.get('MAPE', 0)
                if lookback != 6:
                    diff = mape - baseline_mape
                    if diff > 0:
                        f.write(f"  - Lookback={lookback}: MAPE={mape:.2f}% (+{diff:.2f}%)\n")
                    elif diff < 0:
                        f.write(f"  - Lookback={lookback}: MAPE={mape:.2f}% ({diff:.2f}%) ⭐\n")
                    else:
                        f.write(f"  - Lookback={lookback}: MAPE={mape:.2f}% (相同)\n")
            f.write("\n")
        
        # 详细结果
        f.write("=" * 80 + "\n")
        f.write("详细结果\n")
        f.write("=" * 80 + "\n\n")
        
        for result in all_results:
            lookback = result['Lookback']
            f.write(f"Lookback = {lookback}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  MAPE: {result.get('MAPE', 0):.2f}%\n")
            f.write(f"  MSE:  {result.get('MSE', 0):.6f}\n")
            f.write(f"  MAE:  {result.get('MAE', 0):.6f}\n")
            f.write(f"  有效样本: {result.get('有效样本', 0)}\n")
            f.write(f"  忽略样本: {result.get('忽略样本', 0)}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("结论\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"1. 最佳Lookback值：{best_lookback}（MAPE={best_mape:.2f}%）\n")
        f.write(f"2. 建议使用Lookback={best_lookback}进行后续实验\n")
        f.write(f"3. 如果Lookback={best_lookback}与当前基线（Lookback=6）不同，建议更新基线配置\n")
        f.write("\n")
    
    print(f"结果汇总已保存至: {output_file}")
    print("\n汇总结果预览：")
    print("=" * 80)
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:50]:  # 显示前50行
            print(line, end='')

if __name__ == "__main__":
    main()

