#!/bin/bash
# MoE-Nanjin 基线模型训练脚本（优化版）
# 使用与消融实验完全相同的配置，确保基线模型能达到20%以下

source "$(dirname "$0")/_env.sh"

# 基础配置（与消融实验完全一致）
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
echo "MoE-Nanjin 基线模型训练（优化版）"
echo "=========================================="
echo ""
echo "基线模型配置："
echo "  - 专家配置：1×ShortTermExpert (GRU) + 1×LongTermExpert (Transformer) + 1×DistributionShiftExpert (MLP)"
echo "  - 专家数量：3个"
echo "  - 使用20个变量（from + to）"
echo "  - hidden_dim=512"
echo "  - 与消融实验配置完全一致（确保基线模型也能达到20%以下）"
echo "  - 使用GPU: 3,4,5（避免使用0,1,2）"
echo ""
echo "消融实验结果参考："
echo "  - 移除分布偏移专家(no_statistic): MAPE 20.01%"
echo "  - 移除LongTermExpert:  MAPE 20.30%"
echo "  - 移除ShortTermExpert: MAPE 20.69%"
echo "  - 基线模型（预期）:    MAPE < 20.01%"
echo ""
echo "=========================================="
echo ""

# 训练基线模型
echo "=========================================="
echo "[训练] 基线模型（3个专家）"
echo "=========================================="
python3 src/train.py $BASE_CONFIG \
    --ablation_mode baseline \
    --save_dir ../checkpoints/moe_nanjin

if [ $? -ne 0 ]; then
    echo "[ERROR] 基线模型训练失败！"
    exit 1
fi

echo ""
echo "[完成] 基线模型训练完成！"
echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "模型保存在："
echo "  - ../checkpoints/moe_nanjin/"
echo ""
echo "可以使用以下命令评估："
echo "  python3 src/evaluate.py --checkpoint ../checkpoints/moe_nanjin/best_model_latest.pth"
echo "=========================================="
