#!/usr/bin/env python3
"""在测试集上批量评估目录内所有 checkpoint，选出测试 MAPE 最低者"""

import argparse
import re
from pathlib import Path

import torch

from model import MoENanjin
from data_loader import prepare_data
from evaluate import evaluate


def load_and_eval(ckpt_path, device, data_cache):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    lookback = cfg['lookback']
    key = f"lb{lookback}"

    if key not in data_cache:
        _, _, test_loader, scaler, info = prepare_data(
            data_dir=data_cache['data_dir'],
            target_cols=None,
            lookback=lookback,
            horizon=cfg.get('horizon', 1),
            batch_size=data_cache['batch_size'],
            num_workers=4,
        )
        data_cache[key] = (test_loader, scaler, info)
    test_loader, scaler, info = data_cache[key]

    model = MoENanjin(
        input_dim=cfg['input_dim'],
        output_dim=cfg['output_dim'],
        lookback=lookback,
        num_experts=cfg.get('num_experts', 3),
        hidden_dim=cfg['hidden_dim'],
        dropout=0.15,
        ablation_mode=cfg.get('ablation_mode', 'baseline'),
        use_scene_gating=cfg.get('use_scene_gating', False),
        enhanced_statistic=cfg.get('enhanced_statistic', False),
        scene_dim=cfg.get('scene_dim', 0),
    ).to(device)

    state = ckpt['model_state_dict']
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state)

    results, *_ = evaluate(model, test_loader, device, scaler=scaler)
    epoch = ckpt.get('epoch', '?')
    val_mape = ckpt.get('val_mape')
    return {
        'file': ckpt_path.name,
        'path': str(ckpt_path.resolve()),
        'epoch': epoch,
        'val_mape': val_mape,
        'test_mape': results['mape'],
        'test_mse': results['mse'],
        'test_mae': results['mae'],
    }


def collect_ckpts(ckpt_dir: Path):
    patterns = [
        'snapshots_for_test/epoch_*.pth',
        'best_model_epoch_*.pth',
        'best_model_latest.pth',
    ]
    seen = set()
    out = []
    for pat in patterns:
        for p in sorted(ckpt_dir.glob(pat)):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--target_mape', type=float, default=19.5,
                        help='低于此值时高亮提示')
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path.cwd() / ckpt_dir

    ckpts = collect_ckpts(ckpt_dir)
    if not ckpts:
        print(f'未找到 checkpoint: {ckpt_dir}')
        return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cache = {'data_dir': args.data_dir, 'batch_size': args.batch_size}
    rows = []

    print(f'共 {len(ckpts)} 个 checkpoint，在测试集上评估...\n')
    for p in ckpts:
        try:
            r = load_and_eval(p, device, cache)
            rows.append(r)
            flag = ' ★' if r['test_mape'] and r['test_mape'] <= args.target_mape else ''
            print(f"  epoch={r['epoch']:>3}  val={r['val_mape']}  test_MAPE={r['test_mape']:.2f}%{flag}  {r['file']}")
        except Exception as e:
            print(f'  [失败] {p.name}: {e}')

    rows = [r for r in rows if r['test_mape'] is not None]
    rows.sort(key=lambda x: x['test_mape'])

    print('\n' + '=' * 70)
    print('测试集 MAPE 排名（前 10）')
    print('=' * 70)
    for i, r in enumerate(rows[:10], 1):
        print(f"{i:2d}. Epoch {r['epoch']:>3}  test={r['test_mape']:.2f}%  val={r['val_mape']}  {r['file']}")

    if rows:
        best = rows[0]
        print('\n' + '=' * 70)
        print(f"🏆 测试集最优: Epoch {best['epoch']}  MAPE = {best['test_mape']:.2f}%")
        print(f"   文件: {best['path']}")
        if best['test_mape'] <= 19.5:
            print('   ✓ 已达到 ~19% 目标，论文可用此 checkpoint')
        else:
            print(f'   距 19.41% 还差 {best["test_mape"] - 19.41:.2f}%，继续训练或扩大 snapshot 范围')
        print('=' * 70)

        out_txt = ckpt_dir / 'test_mape_ranking.txt'
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write('epoch\tval_mape\ttest_mape\tfile\n')
            for r in rows:
                f.write(f"{r['epoch']}\t{r['val_mape']}\t{r['test_mape']:.4f}\t{r['file']}\n")
        print(f'排名已保存: {out_txt}')


if __name__ == '__main__':
    main()
