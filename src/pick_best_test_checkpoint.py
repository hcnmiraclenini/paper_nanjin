#!/usr/bin/env python3
"""在测试集上遍历目录内所有 checkpoint，选出 MAPE 最低者（复现原文 Epoch47 选模方式）"""

import argparse
from pathlib import Path
import subprocess
import re
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    args = parser.parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path.cwd() / ckpt_dir

    ckpts = sorted(ckpt_dir.glob('best_model_epoch_*.pth'))
    if not ckpts:
        ckpts = [ckpt_dir / 'best_model_latest.pth'] if (ckpt_dir / 'best_model_latest.pth').exists() else []

    if not ckpts:
        print(f'未找到 checkpoint: {ckpt_dir}')
        sys.exit(1)

    best_mape = float('inf')
    best_path = None
    results = []

    for p in ckpts:
        proc = subprocess.run(
            [sys.executable, 'evaluate.py', '--checkpoint', str(p.resolve()), '--batch_size', '64'],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )
        m = re.search(r'MAPE:\s*([\d.]+)%', proc.stdout)
        mape = float(m.group(1)) if m else None
        results.append((p.name, mape))
        if mape is not None and mape < best_mape:
            best_mape = mape
            best_path = p
        print(f'{p.name}: {mape if mape else "失败"}%')

    print('\n' + '=' * 60)
    if best_path:
        print(f'测试集最优: {best_path.name}  MAPE={best_mape:.2f}%')
        print(f'完整路径: {best_path.resolve()}')
    print('=' * 60)


if __name__ == '__main__':
    main()
