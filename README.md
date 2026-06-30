# MoE-Nanjin · 场景条件化铁路客流非平稳分解-路由预测

基于 Regime-aware Mixture Routing（RAMR）的南京南站双向日粒度客流预测项目。当前论文叙事聚焦“趋势-周期-扰动”的非平稳分解归纳偏置，并通过场景/状态条件化路由将不同非平稳成分映射到短期依赖、长期依赖与分布偏移专家。

## 目录结构

```
moe_nanjin1/
├── README.md
├── checkpoints -> ../checkpoints    # 模型权重（符号链接，勿删）
├── src/                             # 核心 Python 代码
├── scripts/                         # 训练 / 评估 / 实验脚本
├── paper/                           # 论文与投稿材料
│   ├── 2026-0268-基于多专家融合的铁路客流多尺度预测方法.docx
│   ├── 论文终稿.docx                 # 与上方定稿保持同步的便捷副本
│   ├── figures/                     # 论文配图（PDF/PNG）
│   ├── revision/                    # 退修对照、实验说明、创新点
│   ├── submission/                  # 投稿附件（协议、无作者 PDF 等）
│   └── tools/                       # 排版 / 实验节生成脚本
└── docs/
    ├── CHECKPOINTS.md               # 权重路径与用途说明
    └── experiments/                 # 消融、lookback 等实验指令
```

## 快速开始

```bash
cd moe_nanjin1

# 固定消融套件复核（不做测试集选模）
python3 src/evaluate_fixed_ablation_suite.py \
  --data_dir data \
  --output_dir ../results/fixed_ablation_suite_20260701

# 导出 Strict RAMR-VE 固定配方预测和 bootstrap 置信区间
python3 src/export_fixed_ensemble_predictions.py \
  --data_dir data \
  --weights ../results/strict_no_leakage_ensemble/all_balanced_final_once/selected_weights.csv \
  --summary ../results/strict_no_leakage_ensemble/all_balanced_final_once/ensemble_summary.json \
  --output_dir ../results/strict_no_leakage_ensemble/fixed_recipe_predictions

# 生成论文机制配图
python3 src/plot_paper_figures.py \
  --checkpoint ../checkpoints/paper_experiments/ramr_full_robust/best_model_latest.pth \
  --output_dir ../results/paper_figures_ramr_full

# 最终交付自动核验
python3 scripts/verify_final_deliverables.py

# 现代强基线优势边界审计
python3 scripts/analyze_modern_baseline_margin.py

# 站点方向误差剖面审计
python3 scripts/analyze_target_error_profile.py

# 流量分层误差审计
python3 scripts/analyze_flow_stratified_errors.py

# Regime 门控对齐审计
python3 scripts/analyze_regime_gate_alignment.py
```

所有 `scripts/*.sh` 会自动 `source scripts/_env.sh`，工作目录为项目根目录。

## 最终交付入口

- 论文 Word 定稿：`paper/2026-0268-基于多专家融合的铁路客流多尺度预测方法.docx`
- 论文 Word 便捷副本：`paper/论文终稿.docx`
- 论文 Markdown：`paper/论文终稿.md`
- 正式审稿回复：`审稿意见逐条回复稿.md`、`审稿意见逐条回复稿.docx`
- 逐条修改说明：`专家意见逐条修改说明.md`
- 完成度审计：`docs/experiments/审稿意见完成度审计_20260701.md`
- 最终一致性核验：`docs/experiments/最终一致性核验_20260701.md`
- 自动核验报告：`docs/experiments/最终交付自动核验_20260701.md`
- 现代强基线优势边界：`docs/experiments/现代强基线优势边界审计_20260701.md`
- 站点方向误差剖面：`docs/experiments/站点方向误差剖面审计_20260701.md`
- 流量分层误差审计：`docs/experiments/流量分层误差审计_20260701.md`
- Regime 门控对齐审计：`docs/experiments/Regime门控对齐审计_20260701.md`

## 最终主结果

铁路主数据集使用项目内真实 CSV：`data/from_nj.csv` 与 `data/to_nj.csv`。二者按 `time` 内连接后得到 846 个共同日期和 20 个双向客流目标变量。严格时间顺序划分为训练/验证/测试，`RobustScaler` 仅在训练集拟合。

| 模型 | MAPE/% | MSE | MAE |
| --- | ---: | ---: | ---: |
| MoE-Rail 原论文结果 | 19.41 | 0.037700 | 0.144000 |
| Strict RAMR-VE | **18.5463** | **0.035640** | **0.138028** |

Strict RAMR-VE 的 checkpoint 权重、集成权重和尺度均由验证集确定，测试集仅用于最终一次评估。固定配方预测摘要见 `docs/experiments/artifacts/strict_ramr_ve_fixed_ensemble_summary_20260701.json`。

## 模型权重

**权重保存在 `/root/data3/huangchanni/moe/checkpoints/`，不在本仓库内。**

| 实验 | 路径 | 说明 |
|------|------|------|
| 机制分析主模型 | `paper_experiments/ramr_full_robust/` | RAMR 完整单 checkpoint |
| 固定消融 | `paper_experiments/ablation_*`、`strict_no_leakage/stat_*` | 场景门控 / regime routing / 负载均衡 / 分布特征集合 |
| Strict RAMR-VE 候选 | `strict_no_leakage/*` | 验证集固定集成候选 checkpoint |
| Baseline | `moe_nanjin/` | 完整三专家模型 |
| 组件消融 | `moe_nanjin_no_{statistic,longterm,shortterm}/` | 2024-12 复现实验 |

详见 [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)。

## 论文文件

- **当前定稿**：`paper/2026-0268-基于多专家融合的铁路客流多尺度预测方法.docx`
- **审稿回复**：`审稿意见逐条回复稿.md`、`审稿意见逐条回复稿.docx`
- **逐条修改说明**：`专家意见逐条修改说明.md`
- **排版工具**：`paper/tools/format_tiedao_submission.py`

## 数据

默认数据目录：`data/`（`from_nj.csv` / `to_nj.csv`）
