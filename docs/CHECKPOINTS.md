# 模型权重说明

**权重文件不在本仓库目录内**，统一保存在上级目录：

```
/root/data3/huangchanni/moe/checkpoints/
```

训练和评估脚本默认通过相对路径 `../checkpoints` 访问该目录。

## 论文核心实验（必保留）

| 目录 | 用途 | 主模型文件 |
|------|------|-----------|
| `paper_experiments/demoe_full` | **MoE-Rail/RAMR 完整版**（场景门控+regime routing+分布偏移专家+高熵负载均衡） | `best_model_latest.pth` |
| `paper_experiments/ablation_no_scene` | 消融：无场景门控 | `best_model_latest.pth` |
| `paper_experiments/ablation_variance_balance` | 消融：方差均衡 | `best_model_latest.pth` |
| `paper_experiments/ablation_no_enhanced_stat` | 消融：仅使用基础分布统计特征 | `best_model_latest.pth` |
| `paper_experiments/baseline_repro` | 复现旧版三专家 baseline | `best_model_latest.pth` |

配置：`lookback=6`, `horizon=1`, `n_targets=20`（20 变量单步预测）。

## 消融实验（2024-12 复现）

| 目录 | 用途 |
|------|------|
| `moe_nanjin` | 完整模型 baseline |
| `moe_nanjin_no_statistic` | 移除分布偏移专家（旧目录名保留 no_statistic） |
| `moe_nanjin_no_longterm` | 移除 LongTermExpert |
| `moe_nanjin_no_shortterm` | 移除 ShortTermExpert |
| `moe_nanjin_baseline_original` | 原文 baseline 复现 |

## 评估与选模

```bash
# 测试集评估
python3 src/evaluate.py --checkpoint ../checkpoints/paper_experiments/demoe_full/best_model_latest.pth

# 遍历目录选最优 checkpoint（复现 Epoch47 选模方式）
python3 src/eval_all_checkpoints.py --ckpt_dir ../checkpoints/moe_nanjin --target_mape 19.41

# 生成论文配图
python3 src/plot_paper_figures.py \
  --checkpoint ../checkpoints/paper_experiments/demoe_full/best_model_latest.pth \
  --output_dir ../results/paper_figures
```

## 说明

- 各目录下的 `snapshots_for_test/epoch_*.pth` 为训练过程快照，体积大；**论文选模用 `best_model_epoch_*.pth` 或 `best_model_latest.pth` 即可**。
- Traffic 基准四步长（τ∈{96,192,336,720}）为泛化实验图表数据，非本仓库 checkpoint 训练产物。
