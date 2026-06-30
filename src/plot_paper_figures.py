"""
论文配图脚本（中文标签）
- 图：三类场景下门控权重箱线图（E5）
- 图：STL分解与门控权重对照（E4）
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.tsa.seasonal import STL

from model import MoENanjin
from data_loader import prepare_data, SCENE_DIM

# 中文字体
plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei', 'SimHei', 'Noto Sans CJK SC',
    'Microsoft YaHei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

EXPERT_CN = ['短期专家', '长期专家', '分布偏移专家']
EXPERT_EN_ORDER = ['ShortTerm', 'LongTerm', 'DistributionShift']


def _load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    ablation = cfg.get('ablation_mode', 'baseline')
    model = MoENanjin(
        input_dim=cfg['input_dim'],
        output_dim=cfg['output_dim'],
        lookback=cfg['lookback'],
        num_experts=cfg.get('num_experts', 3),
        hidden_dim=cfg['hidden_dim'],
        dropout=0.1,
        ablation_mode=ablation,
        use_scene_gating=cfg.get('use_scene_gating', True),
        enhanced_statistic=cfg.get('enhanced_statistic', True),
        statistic_feature_set=cfg.get('statistic_feature_set', 'robust'),
        use_regime_routing=cfg.get('use_regime_routing', False),
        regime_dim=cfg.get('regime_dim', 16),
        scene_dim=cfg.get('scene_dim', SCENE_DIM),
    ).to(device)
    state = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def collect_gate_weights(model, test_loader, dataset_info, device):
    """收集门控权重与场景标签"""
    gates = []
    scenes = []
    dates = []
    
    test_dataset = test_loader.dataset
    merged_df = dataset_info.get('merged_df')
    val_end = dataset_info.get('val_end', 0)
    
    idx_ptr = 0
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                bx, by, sc = batch
                sc = sc.to(device)
            else:
                bx, by = batch
                sc = None
            bx = bx.to(device)
            _, _, gw, _ = model(bx, sc)
            gw_np = gw.cpu().numpy()
            B = gw_np.shape[0]
            
            for i in range(B):
                ds_idx = test_dataset.indices[idx_ptr + i]
                sf = test_dataset.scene_features[ds_idx]
                global_idx = test_dataset.scene_offset + ds_idx
                is_holiday = sf[5] > 0.5
                is_weekend = sf[4] > 0.5
                if is_holiday:
                    label = 'holiday'
                elif is_weekend:
                    label = 'weekend'
                else:
                    label = 'weekday'
                scenes.append(label)
                gates.append(gw_np[i])
                if merged_df is not None:
                    dates.append(pd.Timestamp(merged_df['time'].iloc[global_idx]))
            idx_ptr += B
    
    return np.array(gates), scenes, dates


def plot_gate_boxplot(gates, scenes, out_path):
    """图：场景门控权重箱线图"""
    scene_cn = {'weekday': '工作日', 'weekend': '周末', 'holiday': '节假日'}
    scene_order = ['weekday', 'weekend', 'holiday']
    n_exp = gates.shape[1]
    labels = EXPERT_CN[:n_exp] if n_exp == 3 else [f'专家{i+1}' for i in range(n_exp)]
    
    fig, axes = plt.subplots(1, n_exp, figsize=(4 * n_exp, 4.5), sharey=True)
    if n_exp == 1:
        axes = [axes]
    
    colors = ['#4C72B0', '#55A868', '#C44E52']
    for e, ax in enumerate(axes):
        data = []
        for s in scene_order:
            mask = [sc == s for sc in scenes]
            data.append(gates[mask, e] if any(mask) else np.array([]))
        bp = ax.boxplot(
            data, labels=[scene_cn[s] for s in scene_order],
            patch_artist=True, widths=0.55
        )
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.set_title(labels[e], fontsize=13)
        ax.set_ylabel('门控权重' if e == 0 else '')
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('不同日历场景下的专家门控权重分布', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] 已保存: {out_path}')


def plot_stl_gate(gates, dates, merged_df, target_col, out_path, period=7):
    """图：STL分解与门控权重三联图"""
    df = merged_df[['time', target_col]].copy()
    df = df.sort_values('time').reset_index(drop=True)
    series = df[target_col].astype(float).values
    time_vals = df['time'].values
    
    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    
    date_arr = pd.to_datetime(dates)
    gate_df = pd.DataFrame({
        'time': date_arr,
        'g0': gates[:, 0],
        'g1': gates[:, 1] if gates.shape[1] > 1 else 0,
        'g2': gates[:, 2] if gates.shape[1] > 2 else 0,
    })
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    
    axes[0].plot(time_vals, series, color='#333', lw=1.2)
    axes[0].set_ylabel('客流量')
    axes[0].set_title(f'站点客流序列：{target_col.replace("_from_nj", "").replace("_to_nj", "")}', fontsize=12)
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(time_vals, res.trend, color='#C44E52', lw=1.2)
    axes[1].set_ylabel('趋势项 $T_t$')
    axes[1].set_title('STL 趋势分量', fontsize=11)
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(time_vals, res.seasonal, color='#4C72B0', lw=1.0)
    axes[2].set_ylabel('周期项 $S_t$')
    axes[2].set_title('STL 周期分量', fontsize=11)
    axes[2].grid(alpha=0.3)
    
    axes[3].plot(time_vals, res.resid, color='#55A868', lw=0.9, alpha=0.8)
    axes[3].set_ylabel('残差项 $R_t$')
    axes[3].set_title('STL 残差分量', fontsize=11)
    axes[3].grid(alpha=0.3)
    
    # 门控权重（测试集时间点）
    ax2 = axes[0].twinx()
    t_gate = gate_df['time'].values
    ax2.plot(t_gate, gate_df['g0'].values, '--', color='#C44E52', alpha=0.7, label=f'{EXPERT_CN[0]}权重')
    if gates.shape[1] > 1:
        ax2.plot(t_gate, gate_df['g1'].values, '--', color='#4C72B0', alpha=0.7, label=f'{EXPERT_CN[1]}权重')
    if gates.shape[1] > 2:
        ax2.plot(t_gate, gate_df['g2'].values, '--', color='#55A868', alpha=0.7, label=f'{EXPERT_CN[2]}权重')
    ax2.set_ylabel('门控权重')
    ax2.legend(loc='upper right', fontsize=8)
    
    fig.suptitle('时序分解与专家门控权重对照分析', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path.replace('.png', '_stl.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    # 单独门控对照图
    fig2, ax = plt.subplots(figsize=(12, 3.5))
    t_gate = gate_df['time'].values
    ax.plot(t_gate, gate_df['g0'].values, label=EXPERT_CN[0], color='#C44E52')
    if gates.shape[1] > 1:
        ax.plot(t_gate, gate_df['g1'].values, label=EXPERT_CN[1], color='#4C72B0')
    if gates.shape[1] > 2:
        ax.plot(t_gate, gate_df['g2'].values, label=EXPERT_CN[2], color='#55A868')
    ax.set_xlabel('日期')
    ax.set_ylabel('门控权重')
    ax.set_title('测试集专家门控权重时序曲线', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig2.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f'[OK] 已保存: {out_path}')


def plot_volatility_short_gate(gates, dates, merged_df, target_col, out_path):
    """图：局部波动率与短期专家权重关系"""
    if gates.shape[1] < 1 or not dates:
        return {}

    df = merged_df[['time', target_col]].copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    y = df[target_col].astype(float)
    pct_abs = y.pct_change().abs().replace([np.inf, -np.inf], np.nan)
    df['volatility'] = pct_abs.rolling(7, min_periods=2).mean()

    gate_df = pd.DataFrame({
        'time': pd.to_datetime(dates),
        'short_weight': gates[:, 0],
    })
    aligned = pd.merge(gate_df, df[['time', 'volatility']], on='time', how='left').dropna()
    if len(aligned) < 3:
        return {}

    corr, p_value = stats.pearsonr(aligned['volatility'].values, aligned['short_weight'].values)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(aligned['volatility'], aligned['short_weight'], s=28, alpha=0.72, color='#4C72B0')
    z = np.polyfit(aligned['volatility'].values, aligned['short_weight'].values, 1)
    xs = np.linspace(aligned['volatility'].min(), aligned['volatility'].max(), 100)
    ax.plot(xs, z[0] * xs + z[1], color='#C44E52', lw=1.6)
    ax.set_xlabel('滚动波动率')
    ax.set_ylabel('短期专家门控权重')
    ax.set_title('局部波动率与短期专家激活关系', fontsize=13)
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.96, f'r={corr:.3f}, p={p_value:.2e}',
        transform=ax.transAxes, ha='left', va='top',
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': '#BBBBBB'},
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] 已保存: {out_path}')

    return {
        'target_col': target_col,
        'n': int(len(aligned)),
        'pearson_r': float(corr),
        'p_value': float(p_value),
        'short_weight_mean': float(aligned['short_weight'].mean()),
        'volatility_mean': float(aligned['volatility'].mean()),
    }


def ttest_holiday_residual(gates, scenes):
    """节假日 vs 工作日 残差专家权重 t 检验"""
    if gates.shape[1] < 3:
        return None
    hol = gates[[s == 'holiday' for s in scenes], 2]
    wkd = gates[[s == 'weekday' for s in scenes], 2]
    if len(hol) < 5 or len(wkd) < 5:
        return None
    t, p = stats.ttest_ind(hol, wkd, equal_var=False)
    return {'holiday_mean': float(hol.mean()), 'weekday_mean': float(wkd.mean()),
            't_stat': float(t), 'p_value': float(p)}


def main():
    parser = argparse.ArgumentParser(description='生成论文中文配图')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--output_dir', type=str, default='../results/paper_figures')
    parser.add_argument('--target_col', type=str, default=None,
                        help='STL分析用的目标列，默认第一个from变量')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print('加载模型...')
    model, cfg = _load_model(args.checkpoint, device)
    
    print('准备数据...')
    _, _, test_loader, scaler, dataset_info = prepare_data(
        data_dir=args.data_dir,
        target_cols=None,
        lookback=cfg['lookback'],
        horizon=cfg.get('horizon', 1),
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    print('收集门控权重...')
    gates, scenes, dates = collect_gate_weights(model, test_loader, dataset_info, device)
    
    plot_gate_boxplot(gates, scenes, out_dir / '图_场景门控权重箱线图.png')
    
    target_col = args.target_col or dataset_info['target_cols'][0]
    merged_df = dataset_info['merged_df']
    plot_stl_gate(gates, dates, merged_df, target_col, str(out_dir / '图_STL分解与门控权重对照.png'))
    vol_stats = plot_volatility_short_gate(
        gates, dates, merged_df, target_col,
        out_dir / '图_波动率与短期专家权重关系.png',
    )
    
    tt = ttest_holiday_residual(gates, scenes)
    stats_path = out_dir / '门控统计检验.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({'holiday_distribution_shift_test': tt or {}, 'volatility_short_gate': vol_stats}, f, ensure_ascii=False, indent=2)
    if tt:
        print(f"节假日残差专家权重均值={tt['holiday_mean']:.4f}, "
              f"工作日={tt['weekday_mean']:.4f}, p={tt['p_value']:.4e}")
    print(f'[OK] 统计结果: {stats_path}')


if __name__ == '__main__':
    main()
