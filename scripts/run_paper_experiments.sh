#!/bin/bash
# 论文小改实验 — 后台并行训练（铁路学报）
set -e
source "$(dirname "$0")/_env.sh"

LOG_DIR="../logs/paper_experiments"
CKPT_BASE="../checkpoints/paper_experiments"
mkdir -p "$LOG_DIR" "$CKPT_BASE"

# 与原文 baseline 一致的超参
COMMON="--data_dir ../data --lookback 6 --horizon 1 --batch_size 64 \
  --hidden_dim 512 --dropout 0.2 --num_epochs 300 --patience 50 \
  --learning_rate 0.001 --lambda_mape 0.5 --lambda_mae 1.0 --lambda_mse 0.5 \
  --lambda_balance 0.01 --lambda_range 0.1 --num_workers 8 --use_both"

run_train() {
    local name=$1
    local gpu=$2
    shift 2
    local extra_args="$@"
    local save_dir="${CKPT_BASE}/${name}"
    local log_file="${LOG_DIR}/${name}.log"
    echo "[启动] ${name} -> GPU ${gpu}, 日志: ${log_file}"
    CUDA_VISIBLE_DEVICES=${gpu} nohup python3 -u src/train.py \
        ${COMMON} \
        --save_dir "${save_dir}" \
        ${extra_args} \
        > "${log_file}" 2>&1 &
    echo "  PID=$!"
}

echo "=========================================="
echo "论文实验后台训练 — $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# GPU1: DeMoE-Rail 完整版（E1+E2+E3）
run_train "demoe_full" 1 \
    --use_scene_gating --enhanced_statistic --balance_mode entropy

# GPU2: 无场景门控（消融 Table 7）
run_train "ablation_no_scene" 2 \
    --no_scene_gating --enhanced_statistic --balance_mode entropy

# GPU3: 方差负载均衡（消融 Table 5，旧版）
run_train "ablation_variance_balance" 3 \
    --use_scene_gating --enhanced_statistic --balance_mode variance

# GPU4: 无增强统计专家（消融 Table 6）
run_train "ablation_no_enhanced_stat" 4 \
    --use_scene_gating --no_enhanced_statistic --balance_mode entropy

# GPU5: 复现原文三专家 baseline（可选对照）
run_train "baseline_repro" 5 \
    --no_scene_gating --no_enhanced_statistic --balance_mode variance

echo ""
echo "全部任务已提交后台。查看进度:"
echo "  tail -f ${LOG_DIR}/demoe_full.log"
echo ""
echo "训练完成后生成中文配图:"
echo "  python3 src/plot_paper_figures.py --checkpoint ${CKPT_BASE}/demoe_full/best_model_latest.pth"
echo "=========================================="
