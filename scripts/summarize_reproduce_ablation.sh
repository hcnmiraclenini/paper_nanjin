#!/bin/bash
# 在测试集上评估所有快照，对比 2024-12-08 目标结果
source "$(dirname "$0")/_env.sh"

TARGET_BASELINE=19.41
TARGET_NO_STAT=20.01
TARGET_NO_LONG=20.30
TARGET_NO_SHORT=20.69

declare -A DIRS=(
    ["baseline"]="../checkpoints/moe_nanjin"
    ["no_statistic"]="../checkpoints/moe_nanjin_no_statistic"
    ["no_longterm"]="../checkpoints/moe_nanjin_no_longterm"
    ["no_shortterm"]="../checkpoints/moe_nanjin_no_shortterm"
)

declare -A TARGETS=(
    ["baseline"]=$TARGET_BASELINE
    ["no_statistic"]=$TARGET_NO_STAT
    ["no_longterm"]=$TARGET_NO_LONG
    ["no_shortterm"]=$TARGET_NO_SHORT
)

OUT="../results/reproduce_ablation_20241208_summary.txt"
mkdir -p ../results
echo "消融复现 — 测试集选模结果 $(date)" | tee "$OUT"
echo "目标: baseline=${TARGET_BASELINE}% no_stat=${TARGET_NO_STAT}% no_long=${TARGET_NO_LONG}% no_short=${TARGET_NO_SHORT}%" | tee -a "$OUT"
echo "========================================" | tee -a "$OUT"

for mode in baseline no_statistic no_longterm no_shortterm; do
    dir="${DIRS[$mode]}"
    tgt="${TARGETS[$mode]}"
    echo "" | tee -a "$OUT"
    echo ">>> ${mode} (目标 ${tgt}%)" | tee -a "$OUT"
    python3 src/eval_all_checkpoints.py --ckpt_dir "$dir" --target_mape "$tgt" 2>&1 | tee -a "$OUT" | tail -8
done

echo "" | tee -a "$OUT"
echo "完整排名见各目录下 test_mape_ranking.txt" | tee -a "$OUT"
echo "汇总已保存: $OUT"
