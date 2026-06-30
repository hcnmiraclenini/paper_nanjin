#!/bin/bash
# MoE-Nanjin 回看窗口（Lookback）消融实验
# 测试不同的lookback值对模型性能的影响

source "$(dirname "$0")/_env.sh"

# 基础配置（与基线模型完全一致，只改变lookback）
# 参考：README_ABLATION.txt 中的基线模型配置
BASE_CONFIG="--data_dir data \
    --horizon 1 \
    --batch_size 64 \
    --hidden_dim 512 \
    --dropout 0.1 \
    --num_epochs 500 \
    --patience 80 \
    --learning_rate 0.001 \
    --lambda_mape 0.5 \
    --lambda_mae 1.0 \
    --lambda_mse 0.5 \
    --lambda_balance 0.01 \
    --lambda_range 0.1 \
    --use_both \
    --num_workers 8 \
    --gpu_ids 3,4,5"

# 要测试的lookback值
LOOKBACK_VALUES=(6 7 10 14 30)

echo "=========================================="
echo "MoE-Nanjin 回看窗口（Lookback）消融实验"
echo "=========================================="
echo ""
echo "实验配置："
echo "  - 基线模型：3个专家（ShortTerm + LongTerm + Statistic）"
echo "  - 测试lookback值：${LOOKBACK_VALUES[@]}"
echo "  - 其他配置与基线模型完全一致"
echo "  - 使用GPU: 3,4,5"
echo "  - 注意：lookback=30需要更多历史数据，训练样本数会减少"
echo ""
echo "=========================================="
echo ""

# 创建结果目录
RESULTS_DIR="../results/moe_nanjin_lookback_ablation"
mkdir -p "$RESULTS_DIR"

# 记录开始时间
START_TIME=$(date +%s)
echo "实验开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 遍历每个lookback值
for LOOKBACK in "${LOOKBACK_VALUES[@]}"; do
    echo "=========================================="
    echo "[实验] Lookback = $LOOKBACK"
    echo "=========================================="
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 设置保存目录
    SAVE_DIR="../checkpoints/moe_nanjin_lookback_${LOOKBACK}"
    
    # 训练模型
    python3 src/train.py $BASE_CONFIG \
        --lookback $LOOKBACK \
        --ablation_mode baseline \
        --save_dir "$SAVE_DIR"
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Lookback=$LOOKBACK 训练失败！"
        echo ""
        continue
    fi
    
    echo ""
    echo "[完成] Lookback=$LOOKBACK 训练完成！"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 评估模型
    echo "[评估] Lookback=$LOOKBACK 测试集性能..."
    python3 src/evaluate.py \
        --checkpoint "$SAVE_DIR/best_model_latest.pth" \
        --data_dir data \
        --batch_size 64 \
        --num_workers 8 > "$RESULTS_DIR/lookback_${LOOKBACK}_evaluation.txt" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[完成] Lookback=$LOOKBACK 评估完成！"
    else
        echo "[警告] Lookback=$LOOKBACK 评估失败！"
    fi
    echo ""
    
    # 等待一下，避免GPU资源冲突
    sleep 5
done

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "结果保存位置："
for LOOKBACK in "${LOOKBACK_VALUES[@]}"; do
    echo "  - Lookback=$LOOKBACK: ../checkpoints/moe_nanjin_lookback_${LOOKBACK}/"
    echo "  - 评估结果: $RESULTS_DIR/lookback_${LOOKBACK}_evaluation.txt"
done
echo ""
echo "可以使用以下命令查看结果："
echo "  cat $RESULTS_DIR/lookback_*_evaluation.txt"
echo "=========================================="
