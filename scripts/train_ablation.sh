#!/bin/bash
# MoE-Nanjin 消融实验训练脚本
# 运行3个消融实验：
# 1. 移除分布偏移专家（旧消融名 no_statistic）
# 2. 移除LongTermExpert
# 3. 移除ShortTermExpert

source "$(dirname "$0")/_env.sh"

# 基础配置（与原始实验保持一致，MAPE=20%的配置）
# 使用后面的GPU（3,4,5等），避免使用0,1,2
BASE_CONFIG="--data_dir data \
    --lookback 6 \
    --horizon 1 \
    --batch_size 64 \
    --hidden_dim 512 \
    --dropout 0.1 \
    --num_epochs 300 \
    --patience 50 \
    --learning_rate 0.001 \
    --lambda_mape 0.5 \
    --lambda_mae 1.0 \
    --lambda_mse 0.5 \
    --lambda_balance 0.01 \
    --lambda_range 0.1 \
    --use_both \
    --num_workers 8 \
    --gpu_ids 3,4,5"

echo "=========================================="
echo "MoE-Nanjin 消融实验"
echo "=========================================="
echo ""
echo "将运行以下3个消融实验："
echo "  1. 移除分布偏移专家 (no_statistic)"
echo "  2. 移除LongTermExpert (no_longterm)"
echo "  3. 移除ShortTermExpert (no_shortterm)"
echo ""
echo "基础配置："
echo "  - 使用20个变量（from + to）"
echo "  - hidden_dim=512"
echo "  - 与原始实验保持一致（MAPE=20%）"
echo "  - 使用GPU: 3,4,5（避免使用0,1,2）"
echo ""
echo "=========================================="
echo ""

# 实验1：移除分布偏移专家
echo "=========================================="
echo "[实验1/3] 移除分布偏移专家"
echo "=========================================="
python3 src/train.py $BASE_CONFIG \
    --ablation_mode no_statistic \
    --save_dir ../checkpoints/moe_nanjin_no_statistic

if [ $? -ne 0 ]; then
    echo "[ERROR] 实验1失败！"
    exit 1
fi

echo ""
echo "[完成] 实验1完成！"
echo ""

# 实验2：移除LongTermExpert
echo "=========================================="
echo "[实验2/3] 移除LongTermExpert"
echo "=========================================="
python3 src/train.py $BASE_CONFIG \
    --ablation_mode no_longterm \
    --save_dir ../checkpoints/moe_nanjin_no_longterm

if [ $? -ne 0 ]; then
    echo "[ERROR] 实验2失败！"
    exit 1
fi

echo ""
echo "[完成] 实验2完成！"
echo ""

# 实验3：移除ShortTermExpert
echo "=========================================="
echo "[实验3/3] 移除ShortTermExpert"
echo "=========================================="
python3 src/train.py $BASE_CONFIG \
    --ablation_mode no_shortterm \
    --save_dir ../checkpoints/moe_nanjin_no_shortterm

if [ $? -ne 0 ]; then
    echo "[ERROR] 实验3失败！"
    exit 1
fi

echo ""
echo "[完成] 实验3完成！"
echo ""

echo "=========================================="
echo "所有消融实验完成！"
echo "=========================================="
echo ""
echo "实验结果保存在："
echo "  - ../checkpoints/moe_nanjin_no_statistic/"
echo "  - ../checkpoints/moe_nanjin_no_longterm/"
echo "  - ../checkpoints/moe_nanjin_no_shortterm/"
echo ""
echo "可以使用 evaluate.py 评估各实验的测试集性能"
echo "=========================================="
