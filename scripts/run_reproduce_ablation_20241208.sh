#!/bin/bash
# 复现 2024-12-08 消融实验（与消融实验报告超参一致；选模按验证集）
set -e
source "$(dirname "$0")/_env.sh"

LOG_DIR="../logs/reproduce_ablation_20241208"
mkdir -p "$LOG_DIR"

# 消融实验报告统一配置（dropout=0.1, lr=0.001）
COMMON="--data_dir data --lookback 6 --horizon 1 --batch_size 64 \
  --hidden_dim 512 --dropout 0.1 --num_epochs 300 --patience 50 \
  --learning_rate 0.001 --weight_decay 0.0001 \
  --lambda_mape 0.5 --lambda_mae 1.0 --lambda_mse 0.5 \
  --lambda_balance 0.01 --lambda_range 0.1 \
  --use_both --num_workers 8 --seed 42 \
  --no_scene_gating --no_enhanced_statistic --balance_mode variance \
  --snapshot_every 5 --max_keep_models 30"

run_one() {
    local mode=$1
    local save=$2
    local gpu=$3
    local log="${LOG_DIR}/${mode}.log"
    echo "[启动] ${mode} -> GPU${gpu} -> ${save}"
    CUDA_VISIBLE_DEVICES=${gpu} nohup python3 -u src/train.py \
        ${COMMON} \
        --ablation_mode "${mode}" \
        --save_dir "${save}" \
        > "${log}" 2>&1 &
    echo "  PID=$!  log=${log}"
}

# 停掉占用 GPU 的旧 baseline 重训（避免抢卡）
pkill -f "train.py.*moe_nanjin_baseline_original" 2>/dev/null || true
sleep 2

echo "=========================================="
echo "复现 2024-12-08 消融 + 基线 (验证集选模)"
echo "配置: dropout=0.1 lr=0.001 patience=50"
echo "=========================================="

# 清空旧空目录中的残留（保留 snapshots 若有）
for d in moe_nanjin moe_nanjin_no_statistic moe_nanjin_no_longterm moe_nanjin_no_shortterm; do
    mkdir -p "../checkpoints/${d}"
done

run_one "baseline"      "../checkpoints/moe_nanjin"              1
run_one "no_statistic"  "../checkpoints/moe_nanjin_no_statistic" 2
run_one "no_longterm"   "../checkpoints/moe_nanjin_no_longterm"  3
run_one "no_shortterm"  "../checkpoints/moe_nanjin_no_shortterm" 4

echo ""
echo "4 路训练已提交。查看: tail -f ${LOG_DIR}/baseline.log"
echo "训练结束后运行: bash summarize_reproduce_ablation.sh"
