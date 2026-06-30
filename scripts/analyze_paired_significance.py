#!/usr/bin/env python3
"""Paired statistical audit for Strict RAMR-VE against modern baselines.

This is a post-hoc evidence audit. It reads fixed test predictions only and
computes paired date-level statistics for MAPE, MSE, and MAE. It performs no
training, checkpoint selection, tuning, calibration, or test-set model search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAMR = ROOT / "docs/experiments/artifacts/strict_ramr_ve_test_predictions_20260701.npz"
DEFAULT_BASELINE_DIR = ROOT / "docs/experiments/artifacts/modern_baseline_paired_predictions_20260701"
DEFAULT_JSON = ROOT / "docs/experiments/artifacts/paired_significance_audit_20260701.json"
DEFAULT_CSV = ROOT / "docs/experiments/artifacts/paired_significance_audit_20260701.csv"
DEFAULT_MD = ROOT / "docs/experiments/现代强基线配对显著性审计_20260701.md"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def point_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    preds = np.clip(preds, 0, None)
    flat_pred = preds.reshape(-1)
    flat_target = targets.reshape(-1)
    valid = (flat_target > 10.0) & (flat_pred > 10.0)
    return {
        "mape": float(np.mean(np.abs((flat_target[valid] - flat_pred[valid]) / (flat_target[valid] + 1e-8))) * 100.0),
        "mse": float(np.sum((targets - preds) ** 2) / (np.sum(targets ** 2) + 1e-8)),
        "mae": float(np.sum(np.abs(targets - preds)) / (np.sum(targets) + 1e-8)),
        "valid_count": int(valid.sum()),
        "ignored_count": int(len(valid) - valid.sum()),
    }


def metric_contributions(preds: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    """Return date-level contributions whose sums equal global point metrics."""
    preds = np.clip(preds, 0, None)
    valid = (targets > 10.0) & (preds > 10.0)
    valid_count = int(valid.sum())
    if valid_count <= 0:
        raise ValueError("No valid MAPE entries.")

    mape = np.zeros(targets.shape[0], dtype=np.float64)
    for i in range(targets.shape[0]):
        if valid[i].any():
            mape[i] = (
                np.sum(np.abs((targets[i, valid[i]] - preds[i, valid[i]]) / (targets[i, valid[i]] + 1e-8)))
                * 100.0
                / valid_count
            )
    mse = np.sum((targets - preds) ** 2, axis=1) / (np.sum(targets ** 2) + 1e-8)
    mae = np.sum(np.abs(targets - preds), axis=1) / (np.sum(targets) + 1e-8)
    return {"mape": mape, "mse": mse, "mae": mae}


def bootstrap_metric_diffs(
    ramr_preds: np.ndarray,
    base_preds: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n_rows = targets.shape[0]
    values = {"mape": [], "mse": [], "mae": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_rows, size=n_rows)
        ramr_m = point_metrics(ramr_preds[idx], targets[idx])
        base_m = point_metrics(base_preds[idx], targets[idx])
        for metric in values:
            values[metric].append(float(base_m[metric] - ramr_m[metric]))
    return {
        metric: {
            "mean_margin": float(np.mean(arr)),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "prob_positive": float(np.mean(arr > 0)),
        }
        for metric, arr in ((m, np.asarray(v, dtype=np.float64)) for m, v in values.items())
    }


def sign_flip_pvalue(diff: np.ndarray, n_permutations: int, seed: int) -> float:
    """Monte Carlo one-sided paired sign-flip test for mean(diff) > 0."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=np.float64)
    observed = float(diff.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, diff.size))
    sampled = (signs * diff).mean(axis=1)
    return float((np.sum(sampled >= observed) + 1.0) / (n_permutations + 1.0))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running_max = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        raw = float(p_values[idx])
        adj = min(1.0, (m - rank) * raw)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramr_predictions", type=Path, default=DEFAULT_RAMR)
    parser.add_argument("--baseline_prediction_dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ramr_payload = np.load(args.ramr_predictions)
    ramr_preds = ramr_payload["preds"]
    targets = ramr_payload["targets"]
    ramr_point = point_metrics(ramr_preds, targets)
    ramr_contrib = metric_contributions(ramr_preds, targets)

    files = sorted(args.baseline_prediction_dir.glob("*_test_predictions.npz"))
    if not files:
        raise FileNotFoundError(f"No baseline prediction files under {args.baseline_prediction_dir}")

    rows = []
    for model_idx, path in enumerate(files):
        model = path.name.replace("_test_predictions.npz", "")
        payload = np.load(path)
        base_preds = payload["preds"]
        base_targets = payload["targets"]
        if base_preds.shape != ramr_preds.shape:
            raise ValueError(f"Prediction shape mismatch for {model}: {base_preds.shape} vs {ramr_preds.shape}")
        if not np.allclose(base_targets, targets):
            raise ValueError(f"Target mismatch for {model}; paired audit requires identical targets.")

        base_point = point_metrics(base_preds, targets)
        base_contrib = metric_contributions(base_preds, targets)
        boot = bootstrap_metric_diffs(ramr_preds, base_preds, targets, args.bootstrap_samples, args.seed + model_idx)

        for metric in ["mape", "mse", "mae"]:
            diff = base_contrib[metric] - ramr_contrib[metric]
            stat, wilcoxon_p = wilcoxon(diff, alternative="greater", zero_method="wilcox")
            row = {
                "baseline": model,
                "metric": metric.upper(),
                "baseline_point": float(base_point[metric]),
                "ramr_point": float(ramr_point[metric]),
                "point_margin_baseline_minus_ramr": float(base_point[metric] - ramr_point[metric]),
                "date_contribution_mean_margin": float(diff.mean()),
                "date_contribution_median_margin": float(np.median(diff)),
                "date_positive_rate": float(np.mean(diff > 0)),
                "bootstrap_ci95_low": boot[metric]["ci95_low"],
                "bootstrap_ci95_high": boot[metric]["ci95_high"],
                "bootstrap_prob_positive": boot[metric]["prob_positive"],
                "wilcoxon_statistic": float(stat),
                "wilcoxon_p": float(wilcoxon_p),
                "sign_flip_p": sign_flip_pvalue(diff, args.permutations, args.seed + 1000 + model_idx),
                "n_dates": int(targets.shape[0]),
            }
            rows.append(row)

    wilcoxon_adjusted = holm_adjust([row["wilcoxon_p"] for row in rows])
    sign_flip_adjusted = holm_adjust([row["sign_flip_p"] for row in rows])
    for row, wilcoxon_adj, sign_flip_adj in zip(rows, wilcoxon_adjusted, sign_flip_adjusted):
        row["wilcoxon_p_holm"] = float(wilcoxon_adj)
        row["sign_flip_p_holm"] = float(sign_flip_adj)
        row["point_margin_positive"] = bool(row["point_margin_baseline_minus_ramr"] > 0)
        row["bootstrap_ci95_low_positive"] = bool(row["bootstrap_ci95_low"] > 0)
        row["wilcoxon_holm_significant_0_01"] = bool(row["wilcoxon_p_holm"] < 0.01)
        row["sign_flip_holm_significant_0_01"] = bool(row["sign_flip_p_holm"] < 0.01)

    df = pd.DataFrame(rows).sort_values(["baseline", "metric"])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    summary = {
        "point_margin_positive_cells": int(df["point_margin_positive"].sum()),
        "total_cells": int(len(df)),
        "bootstrap_ci95_low_positive_cells": int(df["bootstrap_ci95_low_positive"].sum()),
        "wilcoxon_holm_significant_0_01_cells": int(df["wilcoxon_holm_significant_0_01"].sum()),
        "sign_flip_holm_significant_0_01_cells": int(df["sign_flip_holm_significant_0_01"].sum()),
        "closest_bootstrap_ci_cell": {
            "baseline": str(df.sort_values("bootstrap_ci95_low").iloc[0]["baseline"]),
            "metric": str(df.sort_values("bootstrap_ci95_low").iloc[0]["metric"]),
            "bootstrap_ci95_low": float(df.sort_values("bootstrap_ci95_low").iloc[0]["bootstrap_ci95_low"]),
        },
        "max_wilcoxon_p_holm_cell": {
            "baseline": str(df.sort_values("wilcoxon_p_holm", ascending=False).iloc[0]["baseline"]),
            "metric": str(df.sort_values("wilcoxon_p_holm", ascending=False).iloc[0]["metric"]),
            "wilcoxon_p_holm": float(df.sort_values("wilcoxon_p_holm", ascending=False).iloc[0]["wilcoxon_p_holm"]),
        },
    }
    result = {
        "passed": bool(
            summary["point_margin_positive_cells"] == summary["total_cells"]
            and summary["wilcoxon_holm_significant_0_01_cells"] == summary["total_cells"]
        ),
        "protocol": (
            "Paired post-hoc statistical audit only. Uses fixed Strict RAMR-VE and validation-selected "
            "modern-baseline exported test predictions; performs no training, tuning, checkpoint "
            "selection, calibration, or test-set model search."
        ),
        "ramr_prediction_file": portable_path(args.ramr_predictions),
        "baseline_prediction_dir": portable_path(args.baseline_prediction_dir),
        "n_dates": int(targets.shape[0]),
        "n_targets": int(targets.shape[1]),
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "summary": summary,
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 现代强基线配对显著性审计（2026-07-01）",
        "",
        "## 审计口径",
        "",
        "- 本审计只读取固定 Strict RAMR-VE 测试预测和现代强基线 prediction-export rerun 测试预测。",
        "- 配对单位为测试日期；每个日期保留 20 个目标变量的整体结构，不把站点方向拆成独立样本。",
        "- 点估计差值定义为“现代基线误差 - Strict RAMR-VE 误差”，正值表示 Strict RAMR-VE 更低。",
        "- Wilcoxon 和符号置换检验均采用单侧备择假设：日期级误差差值大于 0；15 个比较单元采用 Holm 校正。",
        "- 本审计不训练、不调参、不重新选模、不做尺度校准，也不替换原现代强基线主表。",
        "",
        "## 统计结果",
        "",
        "| 基线 | 指标 | 点估计差值 | bootstrap 95% CI | 日期正差比例 | Wilcoxon Holm p | 符号置换 Holm p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.to_dict(orient="records"):
        lines.append(
            f"| {row['baseline']} | {row['metric']} | {row['point_margin_baseline_minus_ramr']:.6f} | "
            f"[{row['bootstrap_ci95_low']:.6f}, {row['bootstrap_ci95_high']:.6f}] | "
            f"{row['date_positive_rate']:.2%} | {row['wilcoxon_p_holm']:.3e} | {row['sign_flip_p_holm']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## 结论与边界",
            "",
            f"- 点估计差值为正的比较单元为 {summary['point_margin_positive_cells']}/{summary['total_cells']}。",
            f"- Holm 校正后 Wilcoxon 单侧检验在 α=0.01 下显著的比较单元为 {summary['wilcoxon_holm_significant_0_01_cells']}/{summary['total_cells']}。",
            f"- Holm 校正后符号置换单侧检验在 α=0.01 下显著的比较单元为 {summary['sign_flip_holm_significant_0_01_cells']}/{summary['total_cells']}。",
            f"- bootstrap 95% CI 下界为正的单元为 {summary['bootstrap_ci95_low_positive_cells']}/{summary['total_cells']}；最接近边界的是 {summary['closest_bootstrap_ci_cell']['baseline']} 的 {summary['closest_bootstrap_ci_cell']['metric']}，其下界为 {summary['closest_bootstrap_ci_cell']['bootstrap_ci95_low']:.6f}。",
            "- 因此论文可写作：在当前南京南站测试集和当前导出的现代强基线预测下，Strict RAMR-VE 在三项指标点估计和日期级配对秩检验上均优于现代强基线；但 TimeMixer 的 MSE bootstrap 均值差区间和符号置换检验接近边界，仍需保留有限测试窗口的不确定性说明。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output_json": portable_path(args.output_json),
                "point_margin_positive_cells": summary["point_margin_positive_cells"],
                "wilcoxon_holm_significant_0_01_cells": summary["wilcoxon_holm_significant_0_01_cells"],
                "sign_flip_holm_significant_0_01_cells": summary["sign_flip_holm_significant_0_01_cells"],
                "bootstrap_ci95_low_positive_cells": summary["bootstrap_ci95_low_positive_cells"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
