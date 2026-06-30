#!/bin/bash
# MoE-Nanjin 统一训练脚本
# 支持不同的配置参数

source "$(dirname "$0")/_env.sh"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 默认配置
CONFIG=${1:-"optimized"}  # optimized 或 20_vars

case $CONFIG in
    "20_vars")
        echo "=========================================="
        echo "开始训练（20个变量，目标MAPE < 20%）"
        echo "=========================================="
        python3 src/train.py \
            --hidden_dim 512 \
            --lambda_mse 1.0 \
            --lambda_range 0.5 \
            --patience 50 \
            --num_epochs 200 \
            --learning_rate 1e-3 \
            --weight_decay 1e-4 \
            --batch_size 64 \
            --num_workers 8
        ;;
    "optimized"|*)
        echo "=========================================="
        echo "开始训练（优化配置）"
        echo "=========================================="
        python3 src/train.py \
            --data_dir data \
            --lookback 6 \
            --horizon 1 \
            --batch_size 64 \
            --hidden_dim 512 \
            --dropout 0.2 \
            --num_epochs 300 \
            --patience 50 \
            --learning_rate 0.001 \
            --lambda_mape 0.5 \
            --lambda_mae 1.0 \
            --lambda_mse 0.5 \
            --lambda_balance 0.01 \
            --lambda_range 0.1 \
            --save_dir ../checkpoints/moe_nanjin_${TIMESTAMP} \
            --use_both
        ;;
esac

echo "=========================================="
echo "训练完成！"
echo "=========================================="
