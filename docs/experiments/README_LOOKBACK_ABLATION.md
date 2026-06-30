# 回看窗口（Lookback）消融实验说明

## 实验目的

测试不同的回看窗口（lookback）值对MoE模型性能的影响，找出最适合南京南高铁客流预测的回看窗口长度。

## 实验设计

### 测试的Lookback值
- 6, 7, 10, 14, 30

### 实验配置
- **模型**：基线模型（3个专家：ShortTerm + LongTerm + Statistic）
- **其他配置**：与基线模型完全一致
- **注意**：lookback=30需要更多历史数据，训练样本数会相应减少
  - horizon = 1
  - batch_size = 64
  - hidden_dim = 512
  - dropout = 0.15
  - num_epochs = 300
  - patience = 50
  - learning_rate = 0.0008
  - weight_decay = 0.0001
  - lambda_mape = 0.5
  - lambda_mae = 1.0
  - lambda_mse = 0.5
  - lambda_balance = 0.01
  - lambda_range = 0.1
  - use_both = True（使用20个变量）

## 运行实验

### 方法1：运行完整实验脚本（推荐）

```bash
cd /root/data/huangchanni/moe_nanjin1
bash train_lookback_ablation.sh
```

这将依次训练所有lookback值的模型，并自动评估。

### 方法2：单独运行某个lookback值

```bash
cd /root/data/huangchanni/moe_nanjin1

# 例如：训练lookback=5的模型
python train.py \
    --data_dir ../data \
    --lookback 5 \
    --horizon 1 \
    --batch_size 64 \
    --hidden_dim 512 \
    --dropout 0.15 \
    --num_epochs 300 \
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
    --gpu_ids 3,4,5 \
    --ablation_mode baseline \
    --save_dir ../checkpoints/moe_nanjin_lookback_5

# 评估模型
python evaluate.py \
    --checkpoint ../checkpoints/moe_nanjin_lookback_5/best_model_latest.pth \
    --data_dir ../data \
    --batch_size 64 \
    --num_workers 8
```

## 结果汇总

实验完成后，运行汇总脚本：

```bash
cd /root/data/huangchanni/moe_nanjin1
python summarize_lookback_results.py
```

汇总结果将保存在：`../results/moe_nanjin_lookback_ablation/lookback_ablation_summary.txt`

## 结果保存位置

- **模型checkpoints**：`../checkpoints/moe_nanjin_lookback_{lookback}/`
- **评估结果**：`../results/moe_nanjin_lookback_ablation/lookback_{lookback}_evaluation.txt`
- **汇总报告**：`../results/moe_nanjin_lookback_ablation/lookback_ablation_summary.txt`

## 预期结果

实验将生成以下信息：
1. 每个lookback值的测试集MAPE、MSE、MAE
2. 性能排序（按MAPE从低到高）
3. 与基线（lookback=6）的对比
4. 最佳lookback值推荐

## 注意事项

1. **训练时间**：每个lookback值约需2-4小时（取决于GPU性能）
2. **GPU使用**：脚本使用GPU 3,4,5，避免占用0,1,2
3. **随机种子**：所有实验使用相同的随机种子（42）确保可复现性
4. **早停机制**：如果验证集MAPE在50个epoch内无改善，将自动停止训练

## 实验完成后

1. 查看汇总报告，找出最佳lookback值
2. 如果最佳lookback值与当前基线（lookback=6）不同，建议更新基线配置
3. 使用最佳lookback值进行后续实验

