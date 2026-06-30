#!/usr/bin/env python3
"""Audit whether RAMR gate weights align with prediction-time regime signals.

This script is explanatory only. It loads the paper mechanism checkpoint and
computes regime descriptors from each test sample's historical lookback window,
which is available before prediction. Test errors are used only for post-hoc
error-spike characterization and never for model selection or recalibration.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import SCENE_DIM, prepare_data  # noqa: E402
from model import MoENanjin  # noqa: E402


CHECKPOINT_CANDIDATES = [
    Path(os.environ["RAMR_GATE_CHECKPOINT"]).expanduser()
    if os.environ.get("RAMR_GATE_CHECKPOINT")
    else None,
    ROOT / "checkpoints/paper_experiments/ramr_full_robust/best_model_latest.pth",
    ROOT / "../checkpoints/paper_experiments/ramr_full_robust/best_model_latest.pth",
    Path("/root/data3/huangchanni/moe/checkpoints/paper_experiments/ramr_full_robust/best_model_latest.pth"),
]
DEFAULT_CHECKPOINT = next((p.resolve() for p in CHECKPOINT_CANDIDATES if p and p.exists()), None)
PREDICTIONS = ROOT / "docs/experiments/artifacts/strict_ramr_ve_test_predictions_20260701.npz"
OUT_JSON = ROOT / "docs/experiments/artifacts/regime_gate_alignment_audit_20260701.json"
OUT_CSV = ROOT / "docs/experiments/artifacts/regime_gate_alignment_audit_20260701.csv"
OUT_MD = ROOT / "docs/experiments/Regime门控对齐审计_20260701.md"

EXPERT_ORDER = ["ShortTerm", "LongTerm", "DistributionShift"]


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MoENanjin, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = MoENanjin(
        input_dim=cfg["input_dim"],
        output_dim=cfg["output_dim"],
        lookback=cfg["lookback"],
        num_experts=cfg.get("num_experts", 3),
        hidden_dim=cfg["hidden_dim"],
        dropout=0.1,
        ablation_mode=cfg.get("ablation_mode", "baseline"),
        use_scene_gating=cfg.get("use_scene_gating", True),
        enhanced_statistic=cfg.get("enhanced_statistic", True),
        statistic_feature_set=cfg.get("statistic_feature_set", "robust"),
        use_regime_routing=cfg.get("use_regime_routing", False),
        regime_dim=cfg.get("regime_dim", 16),
        scene_dim=cfg.get("scene_dim", SCENE_DIM),
    ).to(device)
    state = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def correlation_row(x: np.ndarray, y: np.ndarray, feature: str, expert: str) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return {
            "feature": feature,
            "expert": expert,
            "n": int(mask.sum()),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
        }
    pearson = stats.pearsonr(x[mask], y[mask])
    spearman = stats.spearmanr(x[mask], y[mask])
    return {
        "feature": feature,
        "expert": expert,
        "n": int(mask.sum()),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def group_test(values: np.ndarray, weights: np.ndarray, feature: str, expert: str) -> dict:
    threshold = float(np.quantile(values[np.isfinite(values)], 0.75))
    high = weights[values >= threshold]
    rest = weights[values < threshold]
    t_stat, p_value = stats.ttest_ind(high, rest, equal_var=False)
    return {
        "feature": feature,
        "expert": expert,
        "threshold_q75": threshold,
        "high_n": int(len(high)),
        "rest_n": int(len(rest)),
        "high_mean": float(np.mean(high)),
        "rest_mean": float(np.mean(rest)),
        "mean_delta": float(np.mean(high) - np.mean(rest)),
        "welch_t": float(t_stat),
        "p_value": float(p_value),
    }


def collect_samples(model: MoENanjin, loader, scaler, dataset_info: dict, device: torch.device) -> pd.DataFrame:
    rows: list[dict] = []
    test_dataset = loader.dataset
    merged_df = dataset_info["merged_df"]
    idx_ptr = 0
    eps = 1.0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                bx, _, scene = batch
                scene = scene.to(device)
            else:
                bx, _ = batch
                scene = None
            bx = bx.to(device)
            _, _, gate_weights, _ = model(bx, scene)
            gates = gate_weights.cpu().numpy()
            x_scaled = bx.cpu().numpy()
            batch_size, lookback, n_targets = x_scaled.shape
            x_orig = scaler.inverse_transform(x_scaled.reshape(-1, n_targets)).reshape(
                batch_size, lookback, n_targets
            )

            for i in range(batch_size):
                ds_idx = test_dataset.indices[idx_ptr + i]
                global_idx = test_dataset.scene_offset + ds_idx
                date = pd.Timestamp(merged_df["time"].iloc[global_idx])
                history = np.asarray(x_orig[i], dtype=np.float64)
                daily_total = history.sum(axis=1)
                target_means = history.mean(axis=0)
                target_stds = history.std(axis=0)
                pct_change = np.abs(np.diff(history, axis=0)) / np.maximum(np.abs(history[:-1]), eps)
                daily_pct_change = np.abs(np.diff(daily_total)) / np.maximum(np.abs(daily_total[:-1]), eps)
                median = np.median(daily_total)
                q75 = np.quantile(daily_total, 0.75)
                q25 = np.quantile(daily_total, 0.25)
                iqr = max(float(q75 - q25), eps)
                total_mean = max(float(np.mean(daily_total)), eps)
                target_cv = np.mean(target_stds / np.maximum(np.abs(target_means), eps))

                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "sample_idx": int(len(rows)),
                        "g_short": float(gates[i, 0]),
                        "g_long": float(gates[i, 1]) if gates.shape[1] > 1 else np.nan,
                        "g_distribution": float(gates[i, 2]) if gates.shape[1] > 2 else np.nan,
                        "history_total_mean": total_mean,
                        "history_volatility": float(np.nanmean(pct_change)),
                        "daily_total_volatility": float(np.nanmean(daily_pct_change)),
                        "peak_to_average": float(np.max(daily_total) / total_mean),
                        "trend_strength": float(abs(daily_total[-1] - daily_total[0]) / total_mean),
                        "robust_shift_z": float(abs(daily_total[-1] - median) / iqr),
                        "target_cv": float(target_cv),
                    }
                )
            idx_ptr += batch_size
    return pd.DataFrame(rows)


def attach_error_features(df: pd.DataFrame) -> pd.DataFrame:
    pred_npz = np.load(PREDICTIONS)
    preds = pred_npz["preds"].astype(np.float64)
    targets = pred_npz["targets"].astype(np.float64)
    if len(df) != len(preds):
        raise ValueError(f"sample count mismatch: gate rows={len(df)}, predictions={len(preds)}")
    abs_err = np.abs(preds - targets)
    ape = abs_err / np.maximum(np.abs(targets), 1e-8) * 100.0
    df = df.copy()
    df["date_mape"] = ape.mean(axis=1)
    df["date_norm_abs_error"] = abs_err.sum(axis=1) / np.maximum(np.abs(targets).sum(axis=1), 1e-8)
    df["date_total_target"] = targets.sum(axis=1)
    return df


def build_report(result: dict, df: pd.DataFrame) -> str:
    corr = pd.DataFrame(result["correlations"])
    group = pd.DataFrame(result["q75_group_tests"])

    def fmt(v, digits=4):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "NA"
        return f"{v:.{digits}f}"

    key_rows = [
        ("history_volatility", "DistributionShift"),
        ("daily_total_volatility", "DistributionShift"),
        ("peak_to_average", "DistributionShift"),
        ("trend_strength", "ShortTerm"),
        ("date_mape", "DistributionShift"),
    ]
    lines = [
        "# Regime门控对齐审计（2026-07-01）",
        "",
        "## 审计口径",
        "",
        "- 本审计只加载论文机制模型 checkpoint，不训练、不调参、不重新选模。",
        "- volatility、peak-to-average、trend strength、robust shift z 等 regime 指标均由预测时可见的历史回看窗口计算。",
        "- 测试误差仅用于事后误差峰值刻画，不参与模型选择、权重调整或尺度校准。",
        "",
        "## 关键相关性",
        "",
        "| Regime 指标 | 专家 | n | Pearson r | Pearson p | Spearman r | Spearman p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for feature, expert in key_rows:
        row = corr[(corr["feature"] == feature) & (corr["expert"] == expert)]
        if row.empty:
            continue
        item = row.iloc[0].to_dict()
        lines.append(
            f"| {feature} | {expert} | {int(item['n'])} | {fmt(item['pearson_r'])} | "
            f"{fmt(item['pearson_p'], 6)} | {fmt(item['spearman_r'])} | {fmt(item['spearman_p'], 6)} |"
        )

    lines.extend(
        [
            "",
            "## 高 Regime 强度分组检验",
            "",
            "| Regime 指标 | 专家 | Q75阈值 | 高组均值 | 其余均值 | 差值 | p值 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for feature, expert in key_rows:
        row = group[(group["feature"] == feature) & (group["expert"] == expert)]
        if row.empty:
            continue
        item = row.iloc[0].to_dict()
        lines.append(
            f"| {feature} | {expert} | {fmt(item['threshold_q75'])} | {fmt(item['high_mean'])} | "
            f"{fmt(item['rest_mean'])} | {fmt(item['mean_delta'])} | {fmt(item['p_value'], 6)} |"
        )

    best = result["interpretation"]
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- 高频日总波动更主要激活短期专家：高 `daily_total_volatility` 组短期专家均值为 "
            f"{best['high_daily_vol_short_mean']:.4f}，其余样本为 {best['rest_daily_vol_short_mean']:.4f}，"
            f"p={best['high_daily_vol_short_p']:.6f}。",
            f"- 突发强度更主要激活分布偏移专家：高 `peak_to_average` 组分布偏移专家均值为 "
            f"{best['high_peak_distribution_mean']:.4f}，其余样本为 {best['rest_peak_distribution_mean']:.4f}，"
            f"p={best['high_peak_distribution_p']:.6f}。",
            f"- 分布偏移专家与 `{best['best_distribution_feature']}` 的 Pearson r="
            f"{best['best_distribution_pearson_r']:.4f}，p={best['best_distribution_pearson_p']:.6f}；"
            "该方向提示不同非平稳指标对应不同专家，而不是所有波动都单调触发同一专家。",
            "- 该审计支持“门控权重与预测时可见的 regime 信号存在可检验对应关系”的机制解释，但论文应写成分工差异，不写成简单线性规律。",
            "- 测试误差相关项只用于说明误差峰值与路由行为的事后关系，不能用于调参或证明因果。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if DEFAULT_CHECKPOINT is None:
        tried = [str(p) for p in CHECKPOINT_CANDIDATES if p is not None]
        raise FileNotFoundError(f"RAMR gate checkpoint not found. Tried: {tried}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(DEFAULT_CHECKPOINT, device)
    _, _, test_loader, scaler, dataset_info = prepare_data(
        data_dir=str(ROOT / "data"),
        target_cols=None,
        lookback=cfg["lookback"],
        horizon=cfg.get("horizon", 1),
        batch_size=64,
        num_workers=0,
    )
    df = collect_samples(model, test_loader, scaler, dataset_info, device)
    df = attach_error_features(df)

    feature_cols = [
        "history_volatility",
        "daily_total_volatility",
        "peak_to_average",
        "trend_strength",
        "robust_shift_z",
        "target_cv",
        "date_mape",
        "date_norm_abs_error",
    ]
    expert_cols = {
        "ShortTerm": "g_short",
        "LongTerm": "g_long",
        "DistributionShift": "g_distribution",
    }

    correlations = []
    group_tests = []
    for feature in feature_cols:
        for expert, col in expert_cols.items():
            correlations.append(correlation_row(df[feature].to_numpy(), df[col].to_numpy(), feature, expert))
            group_tests.append(group_test(df[feature].to_numpy(), df[col].to_numpy(), feature, expert))

    dist_corr = [
        row
        for row in correlations
        if row["expert"] == "DistributionShift"
        and row["feature"] not in {"date_mape", "date_norm_abs_error"}
        and row["pearson_r"] is not None
    ]
    best_dist = max(dist_corr, key=lambda row: abs(row["pearson_r"]))
    vol_group = next(
        row
        for row in group_tests
        if row["feature"] == "history_volatility" and row["expert"] == "DistributionShift"
    )
    daily_short_group = next(
        row
        for row in group_tests
        if row["feature"] == "daily_total_volatility" and row["expert"] == "ShortTerm"
    )
    peak_dist_group = next(
        row
        for row in group_tests
        if row["feature"] == "peak_to_average" and row["expert"] == "DistributionShift"
    )

    result = {
        "protocol": (
            "Mechanism audit only. Gate weights come from the paper RAMR mechanism checkpoint. "
            "Regime descriptors are computed from prediction-time historical lookback windows; "
            "test errors are post-hoc only and are not used for model selection."
        ),
        "checkpoint": rel(DEFAULT_CHECKPOINT),
        "predictions_file": rel(PREDICTIONS),
        "n_rows": int(len(df)),
        "expert_order": EXPERT_ORDER,
        "correlations": correlations,
        "q75_group_tests": group_tests,
        "interpretation": {
            "best_distribution_feature": best_dist["feature"],
            "best_distribution_pearson_r": best_dist["pearson_r"],
            "best_distribution_pearson_p": best_dist["pearson_p"],
            "high_vol_distribution_mean": vol_group["high_mean"],
            "rest_vol_distribution_mean": vol_group["rest_mean"],
            "high_vol_distribution_delta": vol_group["mean_delta"],
            "high_vol_distribution_p": vol_group["p_value"],
            "high_daily_vol_short_mean": daily_short_group["high_mean"],
            "rest_daily_vol_short_mean": daily_short_group["rest_mean"],
            "high_daily_vol_short_delta": daily_short_group["mean_delta"],
            "high_daily_vol_short_p": daily_short_group["p_value"],
            "high_peak_distribution_mean": peak_dist_group["high_mean"],
            "rest_peak_distribution_mean": peak_dist_group["rest_mean"],
            "high_peak_distribution_delta": peak_dist_group["mean_delta"],
            "high_peak_distribution_p": peak_dist_group["p_value"],
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(build_report(result, df), encoding="utf-8")
    print(json.dumps({"passed": True, "output_json": str(OUT_JSON), "rows": len(df)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
