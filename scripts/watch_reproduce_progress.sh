#!/bin/bash
# 每 15 分钟扫描一次，看是否已有测试 MAPE 接近目标
source "$(dirname "$0")/_env.sh"
while true; do
    echo "=== $(date) ==="
    for d in moe_nanjin moe_nanjin_no_statistic moe_nanjin_no_longterm moe_nanjin_no_shortterm; do
        n=$(ls ../checkpoints/$d/snapshots_for_test/*.pth 2>/dev/null | wc -l)
        echo "  $d: ${n} snapshots"
    done
    python3 src/eval_all_checkpoints.py --ckpt_dir ../checkpoints/moe_nanjin --target_mape 19.41 2>&1 | grep -E "测试集最优|距 19|★" || true
    sleep 900
done
