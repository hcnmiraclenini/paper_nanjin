#!/bin/bash
# MoE-Nanjin 基线模型继续训练脚本
# 从Epoch 47的checkpoint继续训练，尝试获得更好的效果

source "$(dirname "$0")/_env.sh"

# 基础配置（与消融实验完全一致）
# 使用后面的GPU（3,4,5等），避免使用0,1,2
BASE_CONFIG="--data_dir data \
    --lookback 6 \
    --horizon 1 \
    --batch_size 64 \
    --hidden_dim 512 \
    --dropout 0.15 \
    --num_epochs 400 \
    --patience 50 \
    --learning_rate 0.0008 \
    --weight_decay 0.0001 \
    --lambda_mape 0.5 \
    --lambda_mae 1.0 \
    --lambda_mse 0.5 \
    --lambda_balance 0.01 \
    --lambda_range 0.1 \
    --use_both \
    --num_workers 8 \
    --gpu_ids 6,7"

# 从Epoch 47的checkpoint继续训练
RESUME_CHECKPOINT="/root/data3/huangchanni/moe/checkpoints/moe_nanjin/best_model_epoch_047_mape_33.01.pth"

echo "=========================================="
echo "MoE-Nanjin 基线模型继续训练"
echo "=========================================="
echo ""
echo "继续训练配置："
echo "  - 从checkpoint恢复: $RESUME_CHECKPOINT"
echo "  - 当前最佳: Epoch 47, 验证MAPE 33.01%, 测试MAPE 19.41%"
echo "  - 继续训练到: 400个epoch"
echo "  - 专家配置：1×ShortTermExpert (GRU) + 1×LongTermExpert (Transformer) + 1×DistributionShiftExpert (MLP)"
echo "  - 专家数量：3个"
echo "  - 使用20个变量（from + to）"
echo "  - hidden_dim=512"
echo "  - dropout=0.15（增加正则化）"
echo "  - learning_rate=0.0008（降低学习率）"
echo "  - weight_decay=0.0001（增加权重衰减）"
echo "  - 使用GPU: 6,7（2个GPU）"
echo ""
echo "目标："
echo "  - 仅基于验证集选择checkpoint和训练设置"
echo "  - 测试集只用于最终一次评估，避免测试集泄露"
echo ""
echo "=========================================="
echo ""

# 检查checkpoint是否存在
if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint文件不存在: $RESUME_CHECKPOINT"
    exit 1
fi

# 继续训练基线模型
echo "=========================================="
echo "[训练] 基线模型（从Epoch 47继续）"
echo "=========================================="
python3 src/train.py $BASE_CONFIG \
    --ablation_mode baseline \
    --save_dir ../checkpoints/moe_nanjin \
    --resume "$RESUME_CHECKPOINT"

if [ $? -ne 0 ]; then
    echo "[ERROR] 基线模型继续训练失败！"
    exit 1
fi

echo ""
echo "[完成] 基线模型继续训练完成！"
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
echo ""
echo "或者评估所有保存的checkpoint，找出测试集表现最好的："
echo "  for ckpt in ../checkpoints/moe_nanjin/best_model_epoch_*.pth; do"
echo "    echo \"评估: \$ckpt\""
echo "    python3 src/evaluate.py --checkpoint \$ckpt"
echo "  done"
echo "=========================================="
