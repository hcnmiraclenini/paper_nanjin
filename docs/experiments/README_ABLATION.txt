消融实验使用说明
================

快速开始
--------

运行所有消融实验
----------------

cd /root/data3/huangchanni/moe/moe_nanjin1
bash train_ablation.sh

这将依次运行3个消融实验：
1. 移除DistributionShiftExpert (no_statistic)
2. 移除LongTermExpert (no_longterm)  
3. 移除ShortTermExpert (no_shortterm)

单独运行某个消融实验
--------------------

# 移除DistributionShiftExpert
python train.py --ablation_mode no_statistic --save_dir ../checkpoints/moe_nanjin_no_statistic

# 移除LongTermExpert
python train.py --ablation_mode no_longterm --save_dir ../checkpoints/moe_nanjin_no_longterm

# 移除ShortTermExpert
python train.py --ablation_mode no_shortterm --save_dir ../checkpoints/moe_nanjin_no_shortterm

消融实验说明
------------

基线模型（baseline）
--------------------
- 专家配置：1×ShortTermExpert (GRU) + 1×LongTermExpert (Transformer) + 1×DistributionShiftExpert (MLP)
- 专家总数：3个
- 预期MAPE：~20%（与原始实验一致）

实验1：移除DistributionShiftExpert (no_statistic)
------------------------------------------
- 专家配置：1×ShortTermExpert (GRU) + 1×LongTermExpert (Transformer)
- 专家总数：2个
- 目的：验证分布偏移专家的贡献

实验2：移除LongTermExpert (no_longterm)
----------------------------------------
- 专家配置：1×ShortTermExpert (GRU) + 1×DistributionShiftExpert (MLP)
- 专家总数：2个
- 目的：验证长期依赖专家的贡献

实验3：移除ShortTermExpert (no_shortterm)
------------------------------------------
- 专家配置：1×LongTermExpert (Transformer) + 1×DistributionShiftExpert (MLP)
- 专家总数：2个
- 目的：验证短期依赖专家的贡献

训练参数
--------

所有消融实验使用相同的训练参数（与基线实验一致）：
- --data_dir ../data
- --lookback 6
- --horizon 1
- --batch_size 64
- --hidden_dim 512
- --dropout 0.1
- --num_epochs 500
- --patience 80
- --learning_rate 0.001
- --lambda_mape 0.5
- --lambda_mae 1.0
- --lambda_mse 0.5
- --lambda_balance 0.01
- --lambda_range 0.1
- --use_both (使用20个变量)

结果保存位置
------------

- 基线模型：../checkpoints/moe_nanjin/
- 移除DistributionShiftExpert：../checkpoints/moe_nanjin_no_statistic/
- 移除LongTermExpert：../checkpoints/moe_nanjin_no_longterm/
- 移除ShortTermExpert：../checkpoints/moe_nanjin_no_shortterm/

评估结果
--------

训练完成后，可以使用 evaluate.py 评估各实验的测试集性能：

# 评估基线模型
python evaluate.py --model_path ../checkpoints/moe_nanjin/best_model_latest.pth

# 评估消融实验
python evaluate.py --model_path ../checkpoints/moe_nanjin_no_statistic/best_model_latest.pth
python evaluate.py --model_path ../checkpoints/moe_nanjin_no_longterm/best_model_latest.pth
python evaluate.py --model_path ../checkpoints/moe_nanjin_no_shortterm/best_model_latest.pth

注意事项
--------

1. 所有实验使用相同的随机种子（42）确保可复现性
2. 每个实验会保留最好的5个模型（按验证MAPE排序）
3. 训练时间：每个实验约2-4小时（取决于GPU性能）
4. 建议先运行基线实验确认MAPE达到~20%

