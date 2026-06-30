# MoE-Nanjin · 高铁客流多尺度预测

基于多专家融合（MoE）的南京南高铁客流预测项目。论文定稿见 `paper/`。

## 目录结构

```
moe_nanjin1/
├── README.md
├── checkpoints -> ../checkpoints    # 模型权重（符号链接，勿删）
├── src/                             # 核心 Python 代码
├── scripts/                         # 训练 / 评估 / 实验脚本
├── paper/                           # 论文与投稿材料
│   ├── 基于多专家融合的高铁客流多尺度预测方法-北京交通大学学报-v2.docx
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

# 训练（优化配置）
bash scripts/train.sh

# 论文五组并行实验
bash scripts/run_paper_experiments.sh

# 测试集评估
python3 src/evaluate.py \
  --checkpoint ../checkpoints/paper_experiments/demoe_full/best_model_latest.pth

# 生成论文配图
python3 src/plot_paper_figures.py \
  --checkpoint ../checkpoints/paper_experiments/demoe_full/best_model_latest.pth \
  --output_dir ../results/paper_figures
```

所有 `scripts/*.sh` 会自动 `source scripts/_env.sh`，工作目录为项目根目录。

## 模型权重

**权重保存在 `/root/data/huangchanni/checkpoints/`，不在本仓库内。**

| 实验 | 路径 | 说明 |
|------|------|------|
| 论文主模型 | `paper_experiments/demoe_full/` | DeMoE-Rail 完整版 |
| 消融 | `paper_experiments/ablation_*` | 场景门控 / 均衡 / 统计专家 |
| Baseline | `moe_nanjin/` | 完整三专家模型 |
| 组件消融 | `moe_nanjin_no_{statistic,longterm,shortterm}/` | 2024-12 复现实验 |

详见 [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)。

## 论文文件

- **当前定稿**：`paper/基于多专家融合的高铁客流多尺度预测方法-北京交通大学学报-v2.docx`
- **退修对照**：`paper/revision/`
- **排版工具**：`paper/tools/format_tiedao_submission.py`

## 数据

默认数据目录：`../data/`（`from_nj.csv` / `to_nj.csv`）
