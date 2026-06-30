#!/bin/bash
# 等待 demoe_full 训练产出 checkpoint 后自动生成中文配图
source "$(dirname "$0")/_env.sh"
CKPT="../checkpoints/paper_experiments/demoe_full/best_model_latest.pth"
OUT="../results/paper_figures"
LOG="../logs/paper_experiments/auto_plot.log"

echo "等待 checkpoint: $CKPT" | tee -a "$LOG"
while [ ! -f "$CKPT" ]; do
    sleep 300
    echo "$(date) 仍在等待..." >> "$LOG"
done

echo "$(date) 开始生成配图" | tee -a "$LOG"
python3 src/plot_paper_figures.py \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    >> "$LOG" 2>&1

echo "$(date) 配图完成，目录: $OUT" | tee -a "$LOG"
