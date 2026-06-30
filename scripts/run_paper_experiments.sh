#!/bin/bash
# 论文小改实验 — 后台并行训练（铁路学报）
set -e
source "$(dirname "$0")/_env.sh"

LOG_DIR="../logs/paper_experiments"
CKPT_BASE="../checkpoints/paper_experiments"
DATA_DIR="${DATA_DIR:-${MOE_NANJIN_ROOT}/data}"
mkdir -p "$LOG_DIR" "$CKPT_BASE"

# 与原文 baseline 一致的超参
COMMON="--data_dir ${DATA_DIR} --lookback 6 --horizon 1 --batch_size 64 \
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
    nohup bash -lc "cd '${MOE_NANJIN_ROOT}' && CUDA_VISIBLE_DEVICES=${gpu} python3 -u src/train.py ${COMMON} --save_dir '${save_dir}' ${extra_args}" \
        > "${log_file}" 2>&1 &
    echo "  PID=$!"
}

echo "=========================================="
echo "论文实验后台训练 — $(date '+%Y-%m-%d %H:%M:%S')"
echo "数据目录: ${DATA_DIR}"
echo "=========================================="

# GPU0: RAMR 完整版（scene + regime + robust distribution shift expert）
run_train "ramr_full_robust" 0 \
    --use_scene_gating --use_regime_routing --enhanced_statistic \
    --statistic_feature_set robust --balance_mode entropy

# GPU1: 无 regime routing（创新点消融）
run_train "ablation_no_regime" 1 \
    --use_scene_gating --no_regime_routing --enhanced_statistic \
    --statistic_feature_set robust --balance_mode entropy

# GPU2: 无场景门控（场景条件化消融）
run_train "ablation_no_scene" 2 \
    --no_scene_gating --use_regime_routing --enhanced_statistic \
    --statistic_feature_set robust --balance_mode entropy

# GPU3: 统计专家仅 mean/std/max
run_train "ablation_stat_basic" 3 \
    --use_scene_gating --use_regime_routing --no_enhanced_statistic \
    --statistic_feature_set basic --balance_mode entropy

# GPU4: 统计专家加入分位数
run_train "ablation_stat_quantile" 4 \
    --use_scene_gating --use_regime_routing --enhanced_statistic \
    --statistic_feature_set quantile --balance_mode entropy

# GPU5: 方差负载均衡旧版（均衡正则消融）
run_train "ablation_variance_balance" 5 \
    --use_scene_gating --use_regime_routing --enhanced_statistic \
    --statistic_feature_set robust --balance_mode variance

echo ""
echo "全部任务已提交后台。查看进度:"
echo "  tail -f ${LOG_DIR}/ramr_full_robust.log"
echo ""
echo "训练完成后生成中文配图:"
echo "  python3 src/plot_paper_figures.py --checkpoint ${CKPT_BASE}/ramr_full_robust/best_model_latest.pth --data_dir ${DATA_DIR}"
echo "=========================================="
