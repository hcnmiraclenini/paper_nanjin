#!/bin/bash
# 立即在测试集上评估已有 checkpoint
source "$(dirname "$0")/_env.sh"
DIR="${1:-../checkpoints/moe_nanjin_baseline_original}"
python3 src/eval_all_checkpoints.py --ckpt_dir "$DIR" --target_mape 19.41
