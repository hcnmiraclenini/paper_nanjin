# 回看窗口（Lookback）消融实验 - 训练指令

## 📋 实验配置

**测试的 Lookback 值**：6, 7, 10, 14, 30

**配置说明**：除 lookback 外，所有参数与基线模型完全一致（参考 README_ABLATION.txt）

## 🚀 训练指令

### Lookback = 6

```bash
cd /root/data3/huangchanni/moe/moe_nanjin1

python3 train.py \
    --data_dir ../data \
    --lookback 6 \
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
    --gpu_ids 3,4,5 \
    --ablation_mode baseline \
    --save_dir ../checkpoints/moe_nanjin_lookback_6
```

### Lookback = 7

```bash
cd /root/data3/huangchanni/moe/moe_nanjin1

python3 train.py \
    --data_dir ../data \
    --lookback 7 \
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
    --gpu_ids 3,4,5 \
    --ablation_mode baseline \
    --save_dir ../checkpoints/moe_nanjin_lookback_7
```

### Lookback = 10

```bash
cd /root/data3/huangchanni/moe/moe_nanjin1

python3 train.py \
    --data_dir ../data \
    --lookback 10 \
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
    --gpu_ids 3,4,5 \
    --ablation_mode baseline \
    --save_dir ../checkpoints/moe_nanjin_lookback_10
```

### Lookback = 14

```bash
cd /root/data3/huangchanni/moe/moe_nanjin1

python3 train.py \
    --data_dir ../data \
    --lookback 14 \
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
    --gpu_ids 3,4,5 \
    --ablation_mode baseline \
    --save_dir ../checkpoints/moe_nanjin_lookback_14
```

### Lookback = 30

```bash
cd /root/data3/huangchanni/moe/moe_nanjin1

python3 train.py \
    --data_dir ../data \
    --lookback 30 \
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
    --gpu_ids 3,4,5 \
    --ablation_mode baseline \
    --save_dir ../checkpoints/moe_nanjin_lookback_30
```

## 📊 评估指令

训练完成后，评估每个模型：

```bash
cd /root/data3/huangchanni/moe/moe_nanjin1

# 创建结果目录
mkdir -p ../results/moe_nanjin_lookback_ablation

# 评估每个 lookback 值
for lookback in 6 7 10 14 30; do
    echo "评估 Lookback = $lookback"
    python3 evaluate.py \
        --checkpoint ../checkpoints/moe_nanjin_lookback_${lookback}/best_model_latest.pth \
        --data_dir ../data \
        --batch_size 64 \
        --num_workers 8 > ../results/moe_nanjin_lookback_ablation/lookback_${lookback}_evaluation.txt 2>&1
    echo "完成 Lookback = $lookback"
    echo ""
done
```

## 📈 汇总结果

```bash
cd /root/data3/huangchanni/moe/moe_nanjin1
python3 summarize_lookback_results.py
```

查看汇总报告：
```bash
cat ../results/moe_nanjin_lookback_ablation/lookback_ablation_summary.txt
```

## 📍 结果保存位置

- **模型checkpoints**：`../checkpoints/moe_nanjin_lookback_{lookback}/`
- **评估结果**：`../results/moe_nanjin_lookback_ablation/lookback_{lookback}_evaluation.txt`
- **汇总报告**：`../results/moe_nanjin_lookback_ablation/lookback_ablation_summary.txt`

## ⚠️ 注意事项

1. **配置一致性**：所有参数与基线模型完全一致（参考 README_ABLATION.txt）
2. **GPU使用**：使用GPU 3,4,5，避免与Timer-XL任务冲突
3. **训练时间**：每个 lookback 值约需2-4小时（500 epochs，patience=80）
4. **训练样本数**：lookback=30需要更多历史数据，训练样本数会减少







