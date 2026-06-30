#!/usr/bin/env python3
"""
从Excel文件生成from_nj.csv和to_nj.csv
用于MoE-Nanjin项目的数据预处理
"""

import pandas as pd
from pathlib import Path
import sys

def convert_excel_to_csv(excel_path, output_dir):
    """
    从Excel文件读取数据并生成from_nj.csv和to_nj.csv
    
    Args:
        excel_path: Excel文件路径
        output_dir: 输出CSV文件的目录
    """
    excel_path = Path(excel_path)
    output_dir = Path(output_dir)
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel文件不存在: {excel_path}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"读取Excel文件: {excel_path}")
    
    # 读取Excel文件
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        print("尝试使用openpyxl引擎...")
        try:
            df = pd.read_excel(excel_path, engine='openpyxl')
        except Exception as e2:
            print(f"使用openpyxl引擎也失败: {e2}")
            raise
    
    print(f"Excel文件形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查是否有time列
    time_cols = [col for col in df.columns if 'time' in col.lower() or '日期' in col or 'date' in col.lower()]
    if not time_cols:
        # 如果没有time列，检查第一列是否是日期
        first_col = df.columns[0]
        if pd.api.types.is_datetime64_any_dtype(df[first_col]) or 'time' in first_col.lower() or '日期' in first_col or 'date' in first_col.lower():
            time_col = first_col
        else:
            raise ValueError("找不到时间列，请确保Excel文件包含'time'、'日期'或'date'列")
    else:
        time_col = time_cols[0]
    
    print(f"使用时间列: {time_col}")
    
    # 站点列表（根据项目总结文档）
    sites = ['北京南', '成都东', '广州南', '汉口', '杭州东', '杭州西', '上海', '上海虹桥', '武汉', '西安北']
    
    # 构建from_nj.csv（南京南→各站点）
    from_cols = [time_col]
    from_data = {}
    from_data[time_col] = df[time_col]
    
    for site in sites:
        # 查找包含该站点名称的列（可能是"站点_from"或"站点_from_nj"等格式）
        possible_cols = [col for col in df.columns if site in col and ('from' in col.lower() or '出' in col or '→' in col)]
        if possible_cols:
            from_data[site] = df[possible_cols[0]]
            from_cols.append(site)
        else:
            print(f"警告: 找不到站点 {site} 的from数据列")
    
    from_df = pd.DataFrame(from_data)
    from_df = from_df.rename(columns={time_col: 'time'})
    
    # 确保time列是datetime类型
    from_df['time'] = pd.to_datetime(from_df['time'])
    from_df = from_df.sort_values('time').reset_index(drop=True)
    
    # 构建to_nj.csv（各站点→南京南）
    to_cols = [time_col]
    to_data = {}
    to_data[time_col] = df[time_col]
    
    for site in sites:
        # 查找包含该站点名称的列（可能是"站点_to"或"站点_to_nj"等格式）
        possible_cols = [col for col in df.columns if site in col and ('to' in col.lower() or '进' in col or '←' in col)]
        if possible_cols:
            to_data[site] = df[possible_cols[0]]
            to_cols.append(site)
        else:
            print(f"警告: 找不到站点 {site} 的to数据列")
    
    to_df = pd.DataFrame(to_data)
    to_df = to_df.rename(columns={time_col: 'time'})
    
    # 确保time列是datetime类型
    to_df['time'] = pd.to_datetime(to_df['time'])
    to_df = to_df.sort_values('time').reset_index(drop=True)
    
    # 保存CSV文件（使用GBK编码，与data_loader.py保持一致）
    from_path = output_dir / 'from_nj.csv'
    to_path = output_dir / 'to_nj.csv'
    
    try:
        from_df.to_csv(from_path, index=False, encoding='GBK')
        print(f"\n已保存: {from_path}")
        print(f"  行数: {len(from_df)}")
        print(f"  列数: {len(from_df.columns)}")
    except Exception as e:
        print(f"使用GBK编码保存失败: {e}")
        print("尝试使用UTF-8编码...")
        from_df.to_csv(from_path, index=False, encoding='utf-8-sig')
        print(f"已保存: {from_path} (UTF-8编码)")
    
    try:
        to_df.to_csv(to_path, index=False, encoding='GBK')
        print(f"\n已保存: {to_path}")
        print(f"  行数: {len(to_df)}")
        print(f"  列数: {len(to_df.columns)}")
    except Exception as e:
        print(f"使用GBK编码保存失败: {e}")
        print("尝试使用UTF-8编码...")
        to_df.to_csv(to_path, index=False, encoding='utf-8-sig')
        print(f"已保存: {to_path} (UTF-8编码)")
    
    print(f"\n转换完成！")
    print(f"输出目录: {output_dir}")
    return from_path, to_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python convert_excel_to_csv.py <excel_path> [output_dir]")
        print("示例: python convert_excel_to_csv.py ../data/2023_2025Updated.xlsx ../data")
        sys.exit(1)
    
    excel_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '../data'
    
    try:
        convert_excel_to_csv(excel_path, output_dir)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

