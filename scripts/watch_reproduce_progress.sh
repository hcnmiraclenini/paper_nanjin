#!/bin/bash
# 每 15 分钟扫描一次验证集排名，不访问测试集
source "$(dirname "$0")/_env.sh"
while true; do
    echo "=== $(date) ==="
    for d in moe_nanjin moe_nanjin_no_statistic moe_nanjin_no_longterm moe_nanjin_no_shortterm; do
        n_old=$(ls ../checkpoints/$d/snapshots_for_test/*.pth 2>/dev/null | wc -l)
        n_new=$(ls ../checkpoints/$d/snapshots_for_diagnostics/*.pth 2>/dev/null | wc -l)
        echo "  $d: $((n_old + n_new)) diagnostic snapshots"
    done
    python3 src/eval_all_checkpoints.py --ckpt_dir ../checkpoints/moe_nanjin --target_mape 19.41 2>&1 | grep -E "验证集最优|val_MAPE|★" || true
    sleep 900
done
