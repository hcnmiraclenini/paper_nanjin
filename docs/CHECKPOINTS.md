# 模型权重说明

**权重文件不在本仓库目录内**，统一保存在上级目录：

```
/root/data3/huangchanni/moe/checkpoints/
```

训练和评估脚本默认通过相对路径 `../checkpoints` 访问该目录。

## 论文核心实验（必保留）

| 目录 | 用途 | 主模型文件 |
|------|------|-----------|
| `paper_experiments/ramr_full_robust` | **RAMR 完整单 checkpoint**（场景门控+regime routing+分布偏移专家+高熵负载均衡），用于机制分析和门控图 | `best_model_latest.pth` |
| `paper_experiments/ablation_no_scene` | 消融：无场景门控 | `best_model_latest.pth` |
| `paper_experiments/ablation_no_regime` | 消融：无 regime routing | `best_model_latest.pth` |
| `paper_experiments/ablation_variance_balance` | 消融：方差均衡 | `best_model_latest.pth` |
| `strict_no_leakage/stat_basic` | 消融：basic 分布统计特征 | `best_model_latest.pth` |
| `strict_no_leakage/stat_quantile` | 消融：quantile 分布统计特征 | `best_model_latest.pth` |
| `strict_no_leakage/mae_robust` | MAE-aware robust 候选 | `best_model_latest.pth` |
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
# 固定消融套件复核（不进行测试集选模）
python3 src/evaluate_fixed_ablation_suite.py \
  --data_dir data \
  --output_dir ../results/fixed_ablation_suite_20260701

# 遍历目录按验证集 MAPE 排名 checkpoint（不访问测试集）
python3 src/eval_all_checkpoints.py --ckpt_dir ../checkpoints/moe_nanjin --target_mape 19.41

# 生成论文配图
python3 src/plot_paper_figures.py \
  --checkpoint ../checkpoints/paper_experiments/ramr_full_robust/best_model_latest.pth \
  --output_dir ../results/paper_figures_ramr_full
```

## 说明

- 新训练会将过程快照写入 `snapshots_for_diagnostics/epoch_*.pth`；历史遗留的 `snapshots_for_test/epoch_*.pth` 仅为旧命名兼容。**论文选模必须依据验证集指标，测试集只用于最终一次评估**。
- Strict RAMR-VE 最终结果来自 `../results/strict_no_leakage_ensemble/all_balanced_final_once/ensemble_summary.json`，对应测试 MAPE=18.5463%、MSE=0.035640、MAE=0.138028。
- Traffic 基准四步长（τ∈{96,192,336,720}）为跨粒度结构鲁棒性补充图表数据，当前仓库未包含可从原始 Traffic 数据完整重训的文件。
