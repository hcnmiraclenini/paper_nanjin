#!/usr/bin/env python3
"""Legacy test-set checkpoint scan.

This script is not leakage-safe and must not be used for paper model
selection. It is retained only for reproducing historical results.
"""

import argparse
from pathlib import Path
import subprocess
import re
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--allow_test_selection', action='store_true',
                        help='LEGACY/DEBUG only: explicitly allow ranking checkpoints by test-set MAPE.')
    args = parser.parse_args()
    if not args.allow_test_selection:
        print(
            '拒绝运行：按测试集选择 checkpoint 会造成测试集泄露，不能用于论文选模。\n'
            '请改用 `python3 src/eval_all_checkpoints.py --ckpt_dir <dir>` 按验证集排名；'
            '测试集只应在最终方案固定后评估一次。\n'
            '若仅为历史复现，请显式追加 --allow_test_selection。'
        )
        sys.exit(2)
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
        print(f'LEGACY 测试集最优: {best_path.name}  MAPE={best_mape:.2f}%')
        print(f'完整路径: {best_path.resolve()}')
        print('警告：该结果只能用于历史复现/诊断，不能用于论文选模。')
    print('=' * 60)


if __name__ == '__main__':
    main()
