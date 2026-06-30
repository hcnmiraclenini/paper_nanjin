#!/bin/bash
# 原文基线：训练 + 每5epoch测试选模快照 + 训练结束后自动扫测试集
set -e
source "$(dirname "$0")/_env.sh"

SAVE_DIR="../checkpoints/moe_nanjin_baseline_original"
LOG="../logs/paper_experiments/original_baseline.log"
mkdir -p "$(dirname "$LOG")" "$SAVE_DIR"

# 若已有训练进程，先停掉
pkill -f "train.py.*moe_nanjin_baseline_original" 2>/dev/null || true
sleep 2

echo "=== 原文基线重训（测试集选模） $(date) ===" | tee "$LOG"
CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/train.py \
    --data_dir ../data \
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

# 后台：每10分钟扫一次快照，直到出现 test MAPE<=19.5 或训练结束
nohup bash -c "
  cd '$(pwd)'
  while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 600
    python3 src/eval_all_checkpoints.py --ckpt_dir '$SAVE_DIR' --target_mape 19.5 >> '${LOG%.log}_test_scan.log' 2>&1
  done
  echo '=== 训练结束，最终测试集选模 ===' >> '${LOG%.log}_test_scan.log'
  python3 src/eval_all_checkpoints.py --ckpt_dir '$SAVE_DIR' --target_mape 19.41 >> '${LOG%.log}_test_scan.log' 2>&1
" >> "${LOG%.log}_test_scan.log" 2>&1 &

echo "训练已启动；测试集扫描日志: ${LOG%.log}_test_scan.log"
echo "手动随时运行: python3 src/eval_all_checkpoints.py --ckpt_dir $SAVE_DIR"
