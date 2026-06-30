"""
MoE-Nanjin 预测结果可视化脚本
根据 test_predictions_comparison.csv 绘制每个变量的预测值与真实值对比曲线图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import argparse

# 设置中文字体 - 直接使用字体文件
font_paths = [
    '/root/.fonts/wqy-microhei.ttc',  # 文泉驿微米黑
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    '/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf',
]

# 找到可用的字体文件
font_path = None
for fp in font_paths:
    if Path(fp).exists():
        font_path = fp
        break

if font_path:
    # 添加字体并设置为默认
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    print(f"使用字体: {font_path}")
else:
    print("警告: 未找到中文字体文件")

plt.rcParams['axes.unicode_minus'] = False


def calculate_mape(y_true, y_pred, threshold=10.0):
    """计算单个变量的MAPE（带阈值过滤）"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # 裁剪预测值为非负
    y_pred = np.clip(y_pred, 0, None)
    
    # 阈值过滤
    mask = (y_true > threshold) & (y_pred > threshold)
    
    if mask.sum() == 0:
        # 如果没有有效样本，返回None
        return None, 0
    
    mape = np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-8)).mean() * 100
    return mape, mask.sum()


def get_display_name(col_name):
    """
    将变量名转换为显示名称
    北京南_from_nj -> 南京南→北京南
    北京南_to_nj -> 北京南→南京南
    """
    if '_from_nj' in col_name:
        site = col_name.replace('_from_nj', '')
        return f"南京南→{site}"
    elif '_to_nj' in col_name:
        site = col_name.replace('_to_nj', '')
        return f"{site}→南京南"
    else:
        return col_name


def get_safe_filename(display_name):
    """将显示名称转换为安全的文件名"""
    # 将箭头替换为下划线
    return display_name.replace('→', '-')


def plot_single_variable(ax, time_index, y_true, y_pred, display_name):
    """绘制单个变量的对比曲线"""
    ax.plot(time_index, y_true, 'b-', linewidth=1.2, label='Groundtruth', alpha=0.8)
    ax.plot(time_index, y_pred, 'r--', linewidth=1.2, label='Prediction', alpha=0.8)
    
    ax.set_xlabel('时间 (样本索引)', fontsize=11)
    ax.set_ylabel('客流量 (人次)', fontsize=11)
    ax.set_title(display_name, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 设置x轴刻度
    n_samples = len(time_index)
    if n_samples > 50:
        step = n_samples // 10
        ax.set_xticks(time_index[::step])


def main():
    parser = argparse.ArgumentParser(description='MoE-Nanjin 预测结果可视化')
    parser.add_argument('--input_file', type=str, 
                        default='../results/moe_nanjin/test_predictions_comparison.csv',
                        help='预测结果CSV文件路径')
    parser.add_argument('--output_dir', type=str,
                        default='../results/moe_nanjin/prediction_plots',
                        help='图片输出目录')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='MAPE计算阈值')
    parser.add_argument('--dpi', type=int, default=150, help='图片DPI')
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 5],
                        help='图片尺寸 (宽 高)')
    
    args = parser.parse_args()
    
    # 解析路径
    input_file = Path(args.input_file)
    if not input_file.is_absolute():
        input_file = Path.cwd() / input_file
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    
    # 检查输入文件
    if not input_file.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        print(f"请先运行 evaluate.py 生成预测结果文件")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取数据
    print(f"读取预测结果: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"  数据形状: {df.shape}")
    
    # 提取变量列表（从列名推断）
    # 列名格式: 变量名_真实值, 变量名_预测值
    all_cols = df.columns.tolist()
    
    # 找出所有变量名
    variable_names = []
    for col in all_cols:
        if col.endswith('_真实值'):
            var_name = col.replace('_真实值', '')
            variable_names.append(var_name)
    
    print(f"  变量数量: {len(variable_names)}")
    print(f"  变量列表: {variable_names[:5]}... (共{len(variable_names)}个)")
    
    # 时间索引（样本索引）
    time_index = np.arange(len(df))
    
    # 统计MAPE结果
    mape_results = []
    
    print(f"\n开始绘图，输出目录: {output_dir}")
    print("=" * 60)
    
    for i, var_name in enumerate(variable_names):
        # 获取真实值和预测值
        true_col = f'{var_name}_真实值'
        pred_col = f'{var_name}_预测值'
        
        y_true = df[true_col].values
        y_pred = df[pred_col].values
        
        # 计算MAPE
        mape, valid_count = calculate_mape(y_true, y_pred, threshold=args.threshold)
        
        # 获取显示名称
        display_name = get_display_name(var_name)
        
        # 记录结果
        mape_results.append({
            'variable': var_name,
            'display_name': display_name,
            'mape': mape,
            'valid_count': valid_count
        })
        
        # 绘图
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
        plot_single_variable(ax, time_index, y_true, y_pred, display_name)
        
        plt.tight_layout()
        
        # 生成文件名
        safe_name = get_safe_filename(display_name)
        if mape is not None:
            filename = f"{safe_name}_MAPE_{mape:.2f}.png"
        else:
            filename = f"{safe_name}_MAPE_NA.png"
        
        # 保存图片
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        # 打印进度
        mape_str = f"{mape:.2f}%" if mape is not None else "N/A"
        print(f"[{i+1:2d}/{len(variable_names)}] {display_name:<20} MAPE={mape_str:<8} -> {filename}")
    
    print("=" * 60)
    print(f"\n[OK] 绘图完成！共生成 {len(variable_names)} 张图片")
    print(f"    保存目录: {output_dir}")
    
    # 输出MAPE汇总
    print(f"\n{'='*60}")
    print("MAPE汇总:")
    print(f"{'='*60}")
    
    valid_mapes = [r['mape'] for r in mape_results if r['mape'] is not None]
    if valid_mapes:
        print(f"  平均MAPE: {np.mean(valid_mapes):.2f}%")
        print(f"  最小MAPE: {np.min(valid_mapes):.2f}%")
        print(f"  最大MAPE: {np.max(valid_mapes):.2f}%")
    
    # 保存MAPE汇总到文件
    mape_df = pd.DataFrame(mape_results)
    mape_file = output_dir / 'mape_summary.csv'
    mape_df.to_csv(mape_file, index=False, encoding='utf-8-sig')
    print(f"\n[OK] MAPE汇总已保存到: {mape_file}")


if __name__ == '__main__':
    main()

