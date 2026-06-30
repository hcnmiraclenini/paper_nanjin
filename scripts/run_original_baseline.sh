#!/bin/bash
# 原文基线：训练 + 每5epoch诊断快照；选模按验证集，测试集仅最终评估一次
set -e
source "$(dirname "$0")/_env.sh"

SAVE_DIR="../checkpoints/moe_nanjin_baseline_original"
LOG="../logs/paper_experiments/original_baseline.log"
RANK_LOG="${LOG%.log}_validation_rank.log"
mkdir -p "$(dirname "$LOG")" "$SAVE_DIR"

# 若已有训练进程，先停掉
pkill -f "train.py.*moe_nanjin_baseline_original" 2>/dev/null || true
sleep 2

echo "=== 原文基线重训（验证集选模） $(date) ===" | tee "$LOG"
CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/train.py \
    --data_dir data \
    --lookback 6 --horizon 1 --batch_size 64 \
    --hidden_dim 512 --dropout 0.15 \
    --num_epochs 300 --patience 80 \
    --learning_rate 0.0008 --weight_decay 0.0001 \
    --lambda_mape 0.5 --lambda_mae 1.0 --lambda_mse 0.5 \
    --lambda_balance 0.01 --lambda_range 0.1 \
    --use_both --num_workers 8 \
    --ablation_mode baseline \
    --no_scene_gating --no_enhanced_statistic --balance_mode variance \
    --snapshot_every 5 --max_keep_models 30 \
    --save_dir "$SAVE_DIR" \
    >> "$LOG" 2>&1 &

TRAIN_PID=$!
echo "训练 PID=$TRAIN_PID" | tee -a "$LOG"

# 后台：每10分钟按验证集排名一次，直到训练结束；不访问测试集
nohup bash -c "
  cd '$(pwd)'
  while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 600
    python3 src/eval_all_checkpoints.py --ckpt_dir '$SAVE_DIR' --target_mape 19.5 >> '$RANK_LOG' 2>&1
  done
  echo '=== 训练结束，最终验证集排名 ===' >> '$RANK_LOG'
  python3 src/eval_all_checkpoints.py --ckpt_dir '$SAVE_DIR' --target_mape 19.41 >> '$RANK_LOG' 2>&1
" >> "$RANK_LOG" 2>&1 &

echo "训练已启动；验证集排名日志: $RANK_LOG"
echo "手动随时运行: python3 src/eval_all_checkpoints.py --ckpt_dir $SAVE_DIR"
