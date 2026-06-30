"""
MoE-Nanjin 数据加载模块
处理from_nj.csv和to_nj.csv，内连接合并，只取共同日期
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

try:
    import chinese_calendar as cc
    HAS_CHINESE_CALENDAR = True
except ImportError:
    HAS_CHINESE_CALENDAR = False

SCENE_DIM = 8  # dow_sin, dow_cos, month_sin, month_cos, is_weekend, is_holiday, is_workday_adj, day_of_month_norm


def build_calendar_features(dates):
    """从日期序列构建场景向量 c_t（用于场景感知门控）"""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    n = len(idx)
    dow = idx.dayofweek.values.astype(np.float32)
    month = idx.month.values.astype(np.float32)
    dom = idx.day.values.astype(np.float32)
    
    dow_sin = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    month_sin = np.sin(2 * np.pi * (month - 1) / 12).astype(np.float32)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12).astype(np.float32)
    is_weekend = (dow >= 5).astype(np.float32)
    dom_norm = (dom / 31.0).astype(np.float32)
    
    is_holiday = np.zeros(n, dtype=np.float32)
    if HAS_CHINESE_CALENDAR:
        for i, d in enumerate(idx):
            try:
                if cc.is_holiday(d.date()):
                    is_holiday[i] = 1.0
            except Exception:
                pass
    
    is_workday_adj = np.zeros(n, dtype=np.float32)
    if HAS_CHINESE_CALENDAR:
        for i, d in enumerate(idx):
            try:
                if dow[i] >= 5 and cc.is_workday(d.date()):
                    is_workday_adj[i] = 1.0
            except Exception:
                pass
    
    scene = np.stack([
        dow_sin, dow_cos, month_sin, month_cos,
        is_weekend, is_holiday, is_workday_adj, dom_norm
    ], axis=1)
    return scene, idx


class NanjinDataset(Dataset):
    """南京南高铁客流数据集类（含日历场景特征）"""
    def __init__(self, data, lookback=4, horizon=1, prev_data=None,
                 scene_features=None, dates=None, scene_offset=0):
        """
        Args:
            data: numpy array [N, n_targets] 已标准化的数据
            scene_features: [N, SCENE_DIM] 日历场景向量
            dates: 与 data 对齐的日期序列（用于可视化）
            scene_offset: scene/dates 相对 data 的全局偏移（验证/测试集用）
        """
        self.data = data
        self.prev_data = prev_data
        self.lookback = lookback
        self.horizon = horizon
        self.n_targets = data.shape[1]
        self.scene_features = scene_features
        self.dates = dates
        self.scene_offset = scene_offset
        
        # 构建有效样本索引
        self.indices = self._build_indices()
        
        prev_info = f", 使用前一个数据集{len(prev_data)}个样本作为lookback" if prev_data is not None else ""
        print(f"数据集初始化完成:")
        print(f"  数据形状: {data.shape}")
        print(f"  目标变量数: {self.n_targets}")
        print(f"  Lookback: {lookback}")
        print(f"  Horizon: {horizon}")
        print(f"  有效样本数: {len(self.indices)} (总数据: {len(data)}, 因horizon={horizon}损失{len(data) - len(self.indices)}个样本{prev_info})")
    
    def _build_indices(self):
        """构建有效的样本索引"""
        indices = []
        total_len = len(self.data)
        
        # 如果有前一个数据集，可以使用它的数据作为lookback
        # 所以当前数据集的所有样本都可以使用（只需要确保有足够的horizon空间）
        # 如果没有前一个数据集，则需要从lookback开始
        start_idx = 0 if self.prev_data is not None else self.lookback
        
        # 确保有足够的horizon空间
        for i in range(start_idx, total_len - self.horizon + 1):
            indices.append(i)
        
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        返回:
            x: [lookback, n_targets] 输入序列
            y: [horizon * n_targets] 目标序列（flatten）
        """
        end_idx = self.indices[idx]
        start_idx = end_idx - self.lookback
        
        # 如果start_idx < 0，说明需要从前一个数据集获取数据
        if start_idx < 0 and self.prev_data is not None:
            # 从前一个数据集获取数据
            prev_start = len(self.prev_data) + start_idx  # start_idx是负数
            prev_end = len(self.prev_data)
            current_start = 0
            current_end = end_idx
            
            # 拼接前一个数据集和当前数据集的数据
            x_prev = self.prev_data[prev_start:prev_end]  # [M, n_targets] where M = -start_idx
            x_curr = self.data[current_start:current_end]  # [N, n_targets] where N = end_idx
            x = np.vstack([x_prev, x_curr])  # [lookback, n_targets]
        else:
            # 正常情况：从当前数据集获取数据
            x = self.data[start_idx:end_idx]  # [lookback, n_targets]
        
        # 未来目标序列（需要预测的）
        pred_end_idx = end_idx + self.horizon
        y = self.data[end_idx:pred_end_idx]  # [horizon, n_targets]
        
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y).flatten()
        
        if self.scene_features is not None:
            scene_c = torch.FloatTensor(self.scene_features[end_idx])
            return x_tensor, y_tensor, scene_c
        
        return x_tensor, y_tensor


def _generate_csv_from_excel(excel_path, output_dir, from_path, to_path):
    """
    从Excel文件生成from_nj.csv和to_nj.csv
    
    Args:
        excel_path: Excel文件路径
        output_dir: 输出目录
        from_path: from_nj.csv输出路径
        to_path: to_nj.csv输出路径
    """
    # 站点列表（根据项目总结文档）
    sites = ['北京南', '成都东', '广州南', '汉口', '杭州东', '杭州西', '上海', '上海虹桥', '武汉', '西安北']
    
    # 读取Excel文件
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
    except:
        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            raise Exception(f"无法读取Excel文件: {e}")
    
    # 查找时间列
    time_cols = [col for col in df.columns if 'time' in str(col).lower() or '日期' in str(col) or 'date' in str(col).lower()]
    if not time_cols:
        # 检查第一列是否是日期
        first_col = df.columns[0]
        if pd.api.types.is_datetime64_any_dtype(df[first_col]):
            time_col = first_col
        else:
            # 尝试将第一列转换为日期
            try:
                df[first_col] = pd.to_datetime(df[first_col])
                time_col = first_col
            except:
                raise ValueError("找不到时间列，请确保Excel文件包含'time'、'日期'或'date'列")
    else:
        time_col = time_cols[0]
    
    # 构建from_nj.csv（南京南→各站点）
    from_data = {'time': pd.to_datetime(df[time_col])}
    
    for site in sites:
        # 查找包含该站点名称的列
        # 可能的列名格式：站点_from, 站点_from_nj, 站点_出, 站点→等
        possible_cols = []
        for col in df.columns:
            col_str = str(col)
            if site in col_str:
                # 检查是否是from相关的列
                if any(keyword in col_str.lower() for keyword in ['from', '出', '→', '->']):
                    possible_cols.append(col)
        
        if possible_cols:
            from_data[site] = df[possible_cols[0]]
        else:
            # 如果找不到，尝试直接使用站点名作为列名
            if site in df.columns:
                from_data[site] = df[site]
            else:
                print(f"  [警告] 找不到站点 {site} 的from数据列，跳过")
    
    from_df = pd.DataFrame(from_data)
    from_df = from_df.sort_values('time').reset_index(drop=True)
    
    # 构建to_nj.csv（各站点→南京南）
    to_data = {'time': pd.to_datetime(df[time_col])}
    
    for site in sites:
        # 查找包含该站点名称的列
        # 可能的列名格式：站点_to, 站点_to_nj, 站点_进, 站点←等
        possible_cols = []
        for col in df.columns:
            col_str = str(col)
            if site in col_str:
                # 检查是否是to相关的列
                if any(keyword in col_str.lower() for keyword in ['to', '进', '←', '<-']):
                    possible_cols.append(col)
        
        if possible_cols:
            to_data[site] = df[possible_cols[0]]
        else:
            # 如果找不到，尝试直接使用站点名作为列名
            if site in df.columns:
                to_data[site] = df[site]
            else:
                print(f"  [警告] 找不到站点 {site} 的to数据列，跳过")
    
    to_df = pd.DataFrame(to_data)
    to_df = to_df.sort_values('time').reset_index(drop=True)
    
    # 保存CSV文件（优先使用GBK编码）
    try:
        from_df.to_csv(from_path, index=False, encoding='GBK')
        to_df.to_csv(to_path, index=False, encoding='GBK')
    except:
        # 如果GBK失败，使用UTF-8
        from_df.to_csv(from_path, index=False, encoding='utf-8-sig')
        to_df.to_csv(to_path, index=False, encoding='utf-8-sig')


def load_and_merge_data(data_dir, from_file='from_nj.csv', to_file='to_nj.csv', use_both=True):
    """
    加载并合并from_nj和to_nj数据（内连接，只取共同日期）
    
    根据项目总结.md的要求：
    - 站点列表：北京南、成都东、广州南、汉口、杭州东、杭州西、上海、上海虹桥、武汉、西安北
    - 内连接合并，只取共同日期（846个）
    - 最终20个变量：10个from变量（站点名_from_nj）+ 10个to变量（站点名_to_nj）
    
    Args:
        data_dir: 数据目录路径
        from_file: from_nj.csv文件名
        to_file: to_nj.csv文件名
        use_both: 是否同时使用from和to（True=20个变量，False=只使用from，10个变量）
    
    Returns:
        merged_df: 合并后的DataFrame（只包含共同日期）
        target_cols: 目标列列表（10个或20个变量）
    """
    # 站点列表（根据项目总结.md）
    sites = ['北京南', '成都东', '广州南', '汉口', '杭州东', '杭州西', '上海', '上海虹桥', '武汉', '西安北']
    
    data_dir = Path(data_dir)
    
    # 解析路径，支持相对路径和绝对路径
    if not data_dir.is_absolute():
        # 相对路径：从当前工作目录解析
        data_dir = Path.cwd() / data_dir
    else:
        data_dir = Path(data_dir)
    
    # 检查数据目录是否存在
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}\n请检查路径是否正确，或使用 --data_dir 参数指定正确的数据目录")
    
    # 检查数据文件是否存在
    from_path = data_dir / from_file
    to_path = data_dir / to_file
    
    if not from_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {from_path}\n请检查文件是否存在，或使用 --data_dir 参数指定正确的数据目录")
    if not to_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {to_path}\n请检查文件是否存在，或使用 --data_dir 参数指定正确的数据目录")
    
    # 读取CSV文件（GBK编码，根据项目总结.md要求）
    print(f"读取数据文件...")
    print(f"  数据目录: {data_dir}")
    try:
        from_df = pd.read_csv(from_path, encoding='GBK')
        to_df = pd.read_csv(to_path, encoding='GBK')
    except UnicodeDecodeError:
        # 如果GBK编码失败，尝试utf-8
        print(f"  [WARNING] GBK编码失败，尝试UTF-8编码...")
        from_df = pd.read_csv(from_path, encoding='utf-8')
        to_df = pd.read_csv(to_path, encoding='utf-8')
    
    print(f"  from_nj.csv: {len(from_df)}行")
    print(f"  to_nj.csv: {len(to_df)}行")
    
    # 确保第一列是time列，并转换为datetime类型
    # 如果列名是乱码，第一列应该是time列
    time_col_from = from_df.columns[0]
    from_df = from_df.rename(columns={time_col_from: 'time'})
    from_df['time'] = pd.to_datetime(from_df['time'])
    
    time_col_to = to_df.columns[0]
    to_df = to_df.rename(columns={time_col_to: 'time'})
    to_df['time'] = pd.to_datetime(to_df['time'])
    
    # 重命名站点列（按顺序对应站点列表）
    # from_nj.csv: 第一列是time，后面10列是站点数据（按顺序对应sites列表）
    # to_nj.csv: 第一列是time，后面10列是站点数据（按顺序对应sites列表）
    # 先保存原始列名，避免重命名过程中列名变化
    from_site_cols = list(from_df.columns[1:])  # 排除time列
    to_site_cols = list(to_df.columns[1:])  # 排除time列
    
    # 重命名from_df的站点列
    rename_dict_from = {}
    for i, site in enumerate(sites):
        if i < len(from_site_cols):
            rename_dict_from[from_site_cols[i]] = site
    from_df = from_df.rename(columns=rename_dict_from)
    
    # 重命名to_df的站点列
    rename_dict_to = {}
    for i, site in enumerate(sites):
        if i < len(to_site_cols):
            rename_dict_to[to_site_cols[i]] = site
    to_df = to_df.rename(columns=rename_dict_to)
    
    # 只保留time列和站点列
    from_df = from_df[['time'] + sites]
    to_df = to_df[['time'] + sites]
    
    # 内连接合并（只取共同日期）
    # 合并时添加后缀区分from和to
    merged_df = pd.merge(from_df, to_df, on='time', how='inner', suffixes=('_from', '_to'))
    merged_df = merged_df.sort_values('time').reset_index(drop=True)
    
    print(f"  合并后: {len(merged_df)}行（共同日期）")
    
    # 根据use_both参数决定使用哪些变量
    target_cols = []
    
    if use_both:
        # 使用20个变量：10个from + 10个to
        # 先添加from变量（南京南→各站点）
        for site in sites:
            from_col = f"{site}_from"  # 南京南→该站点的客流
            if from_col in merged_df.columns:
                new_col_name = f"{site}_from_nj"  # 例如：北京南_from_nj
                merged_df[new_col_name] = merged_df[from_col]
                target_cols.append(new_col_name)
        
        # 再添加to变量（各站点→南京南）
        for site in sites:
            to_col = f"{site}_to"  # 该站点→南京南的客流
            if to_col in merged_df.columns:
                new_col_name = f"{site}_to_nj"  # 例如：北京南_to_nj
                merged_df[new_col_name] = merged_df[to_col]
                target_cols.append(new_col_name)
    else:
        # 只使用10个变量：仅使用from变量（复现之前MAPE=20%的配置）
        print("  [WARNING] 只使用from变量（10个），复现之前MAPE=20%的配置")
        for site in sites:
            from_col = f"{site}_from"  # 南京南→该站点的客流
            if from_col in merged_df.columns:
                new_col_name = f"{site}_from_nj"  # 例如：北京南_from_nj
                merged_df[new_col_name] = merged_df[from_col]
                target_cols.append(new_col_name)
    
    # 只保留time和目标列
    merged_df = merged_df[['time'] + target_cols]
    
    print(f"\n目标列说明:")
    if use_both:
        print(f"  目标变量数: {len(target_cols)}个变量（20个：10个from + 10个to）")
        print(f"  目标列列表:")
        print(f"  【From变量】南京南→各站点（10个）:")
        for i, col in enumerate(target_cols[:10], 1):
            site_name = col.replace('_from_nj', '')
            print(f"    {i:2d}. {col} (南京南 → {site_name})")
        print(f"  【To变量】各站点→南京南（10个）:")
        for i, col in enumerate(target_cols[10:], 1):
            site_name = col.replace('_to_nj', '')
            print(f"    {i+10:2d}. {col} ({site_name} → 南京南)")
    else:
        print(f"  目标变量数: {len(target_cols)}个变量（10个：仅from变量）")
        print(f"  目标列列表:")
        print(f"  【From变量】南京南→各站点（10个）:")
        for i, col in enumerate(target_cols, 1):
            site_name = col.replace('_from_nj', '')
            print(f"    {i:2d}. {col} (南京南 → {site_name})")
    
    return merged_df, target_cols


def prepare_data(
    data_dir,
    target_cols,
    lookback=4,
    horizon=1,
    train_ratio=0.5,
    val_ratio=0.25,
    test_ratio=0.25,
    batch_size=32,
    num_workers=4,
    use_both=True  # 是否同时使用from和to（True=20个变量，False=只使用from，10个变量）
):
    """
    准备数据并创建DataLoader
    
    Args:
        data_dir: 数据目录路径
        target_cols: 目标列列表
        lookback: 回看窗口长度
        horizon: 预测窗口长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        batch_size: batch大小
        num_workers: DataLoader的worker数量
    
    Returns:
        train_loader, val_loader, test_loader, scaler, dataset_info
    """
    # 加载并合并数据
    merged_df, actual_target_cols = load_and_merge_data(data_dir, use_both=use_both)
    
    # 使用实际的目标列
    if target_cols is None:
        target_cols = actual_target_cols
    
    # 日历场景特征（与原始客流同索引）
    all_dates = merged_df['time'].values
    scene_features, date_index = build_calendar_features(all_dates)
    
    target_data = merged_df[target_cols].values
    
    scaler = RobustScaler()
    target_data_scaled = scaler.fit_transform(target_data)
    
    # 按时间顺序划分数据集
    total_len = len(target_data_scaled)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    train_data = target_data_scaled[:train_end]
    val_data = target_data_scaled[train_end:val_end]
    test_data = target_data_scaled[val_end:]
    
    train_scene = scene_features[:train_end]
    val_scene = scene_features[train_end:val_end]
    test_scene = scene_features[val_end:]
    
    train_dates = date_index[:train_end]
    val_dates = date_index[train_end:val_end]
    test_dates = date_index[val_end:]
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(train_data)} ({train_ratio*100:.0f}%)")
    print(f"  验证集: {len(val_data)} ({val_ratio*100:.0f}%)")
    print(f"  测试集: {len(test_data)} ({test_ratio*100:.0f}%)")
    
    # 创建数据集
    # 训练集：没有前一个数据集
    train_dataset = NanjinDataset(
        train_data, lookback=lookback, horizon=horizon, prev_data=None,
        scene_features=train_scene, dates=train_dates, scene_offset=0
    )
    val_dataset = NanjinDataset(
        val_data, lookback=lookback, horizon=horizon,
        prev_data=train_data[-lookback:] if len(train_data) >= lookback else None,
        scene_features=val_scene, dates=val_dates, scene_offset=train_end
    )
    test_dataset = NanjinDataset(
        test_data, lookback=lookback, horizon=horizon,
        prev_data=val_data[-lookback:] if len(val_data) >= lookback else None,
        scene_features=test_scene, dates=test_dates, scene_offset=val_end
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 数据集信息
    dataset_info = {
        'n_targets': len(target_cols),
        'n_exog': 0,  # 无外生变量
        'target_cols': target_cols,
        'exog_cols': [],
        'lookback': lookback,
        'horizon': horizon,
        'input_dim': len(target_cols),  # 输入维度 = 目标变量数（无外生变量）
        'output_dim': len(target_cols) * horizon,
        'scene_dim': SCENE_DIM,
        'merged_df': merged_df,
        'date_index': date_index,
        'train_end': train_end,
        'val_end': val_end,
    }
    
    return train_loader, val_loader, test_loader, scaler, dataset_info

