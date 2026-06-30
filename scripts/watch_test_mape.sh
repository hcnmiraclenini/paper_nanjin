#!/bin/bash
# 立即按验证集 MAPE 排名已有 checkpoint（不访问测试集）
source "$(dirname "$0")/_env.sh"
DIR="${1:-../checkpoints/moe_nanjin_baseline_original}"
python3 src/eval_all_checkpoints.py --ckpt_dir "$DIR" --target_mape 19.41
