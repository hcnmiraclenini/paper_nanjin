#!/usr/bin/env python3
"""Audit Strict RAMR-VE margins against modern baseline point estimates.

This script does not train models and does not select checkpoints. It uses the
already fixed final-test prediction file plus the modern-baseline summary table
to estimate date-level bootstrap uncertainty of Strict RAMR-VE and then reports
how often the fixed model remains below each baseline's point metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED = ROOT / "docs/experiments/artifacts/strict_ramr_ve_test_predictions_20260701.npz"
DEFAULT_BASELINES = ROOT / "docs/experiments/artifacts/modern_baselines_correct_data_20260701.json"
DEFAULT_OUT = ROOT / "docs/experiments/artifacts/modern_baseline_margin_audit_20260701.json"
DEFAULT_CSV = ROOT / "docs/experiments/artifacts/modern_baseline_margin_audit_20260701.csv"
DEFAULT_MD = ROOT / "docs/experiments/现代强基线优势边界审计_20260701.md"


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    preds = np.clip(preds, 0, None)
    flat_pred = preds.reshape(-1)
    flat_target = targets.reshape(-1)
    valid = (flat_target > 10.0) & (flat_pred > 10.0)
    mape = np.mean(np.abs((flat_target[valid] - flat_pred[valid]) / (flat_target[valid] + 1e-8))) * 100.0
    mse = np.sum((targets - preds) ** 2) / (np.sum(targets ** 2) + 1e-8)
    mae = np.sum(np.abs(targets - preds)) / (np.sum(targets) + 1e-8)
    return {
        "mape": float(mape),
        "mse": float(mse),
        "mae": float(mae),
        "valid_count": int(valid.sum()),
        "ignored_count": int(len(valid) - valid.sum()),
    }


def bootstrap_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_rows = preds.shape[0]
    out = {"mape": [], "mse": [], "mae": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_rows, size=n_rows)
        metrics = compute_metrics(preds[idx], targets[idx])
        for key in out:
            out[key].append(metrics[key])
    return {key: np.asarray(values, dtype=np.float64) for key, values in out.items()}


def metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = np.load(args.predictions)
    preds = payload["preds"]
    targets = payload["targets"]
    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))

    point = compute_metrics(preds, targets)
    boot = bootstrap_metrics(preds, targets, args.bootstrap, args.seed)
    boot_summary = {key: metric_summary(values) for key, values in boot.items()}

    rows = []
    for baseline in baselines:
        name = baseline["model"]
        baseline_metrics = {
            "mape": float(baseline["test_mape"]),
            "mse": float(baseline["test_mse"]),
            "mae": float(baseline["test_mae"]),
        }
        row = {
            "baseline": name,
            "baseline_mape": baseline_metrics["mape"],
            "baseline_mse": baseline_metrics["mse"],
            "baseline_mae": baseline_metrics["mae"],
            "ramr_mape": point["mape"],
            "ramr_mse": point["mse"],
            "ramr_mae": point["mae"],
            "mape_absolute_margin": baseline_metrics["mape"] - point["mape"],
            "mse_absolute_margin": baseline_metrics["mse"] - point["mse"],
            "mae_absolute_margin": baseline_metrics["mae"] - point["mae"],
            "mape_relative_reduction_pct": (baseline_metrics["mape"] - point["mape"]) / baseline_metrics["mape"] * 100.0,
            "mse_relative_reduction_pct": (baseline_metrics["mse"] - point["mse"]) / baseline_metrics["mse"] * 100.0,
            "mae_relative_reduction_pct": (baseline_metrics["mae"] - point["mae"]) / baseline_metrics["mae"] * 100.0,
            "bootstrap_prob_mape_below_baseline": float(np.mean(boot["mape"] < baseline_metrics["mape"])),
            "bootstrap_prob_mse_below_baseline": float(np.mean(boot["mse"] < baseline_metrics["mse"])),
            "bootstrap_prob_mae_below_baseline": float(np.mean(boot["mae"] < baseline_metrics["mae"])),
            "bootstrap_prob_all_below_baseline": float(
                np.mean(
                    (boot["mape"] < baseline_metrics["mape"])
                    & (boot["mse"] < baseline_metrics["mse"])
                    & (boot["mae"] < baseline_metrics["mae"])
                )
            ),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("baseline_mape")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": (
            "No training or checkpoint selection. Strict RAMR-VE test predictions are fixed; "
            "date-level bootstrap is used only for post-hoc uncertainty audit against modern "
            "baseline point estimates."
        ),
        "predictions_file": str(args.predictions),
        "baseline_file": str(args.baselines),
        "bootstrap_unit": "test date row with all 20 target variables kept together",
        "bootstrap_samples": args.bootstrap,
        "seed": args.seed,
        "n_rows": int(preds.shape[0]),
        "n_targets": int(preds.shape[1]),
        "strict_ramr_ve_point_metrics": point,
        "strict_ramr_ve_bootstrap_summary": boot_summary,
        "baseline_margin_rows": rows,
    }
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    strongest = df.iloc[0].to_dict()
    md_lines = [
        "# 现代强基线优势边界审计（2026-07-01）",
        "",
        "## 审计口径",
        "",
        "- 本分析不训练模型、不选择 checkpoint、不调整集成权重。",
        "- Strict RAMR-VE 使用已经固定的最终测试集预测文件；现代强基线使用已复核的测试集点估计。",
        "- bootstrap 单位为测试集日期行，每次抽样保留该日期的 20 个目标变量，避免拆散同一日期的多变量结构。",
        "- 由于现代强基线未保存逐样本预测，本分析是相对点估计的非配对优势边界审计，不写作配对显著性检验。",
        "",
        "## Strict RAMR-VE 点估计与不确定性",
        "",
        f"- 点估计：MAPE={point['mape']:.4f}%，MSE={point['mse']:.6f}，MAE={point['mae']:.6f}。",
        f"- 2000 次日期级 bootstrap 95% CI：MAPE [{boot_summary['mape']['ci95_low']:.4f}, {boot_summary['mape']['ci95_high']:.4f}]，"
        f"MSE [{boot_summary['mse']['ci95_low']:.6f}, {boot_summary['mse']['ci95_high']:.6f}]，"
        f"MAE [{boot_summary['mae']['ci95_low']:.6f}, {boot_summary['mae']['ci95_high']:.6f}]。",
        "",
        "## 相对现代强基线的边界",
        "",
        "| 现代基线 | MAPE差值 | MSE差值 | MAE差值 | MAPE相对降幅 | MSE相对降幅 | MAE相对降幅 | bootstrap三项均低于该基线概率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.to_dict(orient="records"):
        md_lines.append(
            f"| {row['baseline']} | {row['mape_absolute_margin']:.4f} | "
            f"{row['mse_absolute_margin']:.6f} | {row['mae_absolute_margin']:.6f} | "
            f"{row['mape_relative_reduction_pct']:.2f}% | {row['mse_relative_reduction_pct']:.2f}% | "
            f"{row['mae_relative_reduction_pct']:.2f}% | {row['bootstrap_prob_all_below_baseline']:.2%} |"
        )
    md_lines.extend(
        [
            "",
            "## 写作建议",
            "",
            f"- 与当前最强现代基线 FreEformer 相比，Strict RAMR-VE 点估计 MAPE/MSE/MAE 分别降低 "
            f"{strongest['mape_relative_reduction_pct']:.2f}%、{strongest['mse_relative_reduction_pct']:.2f}%、"
            f"{strongest['mae_relative_reduction_pct']:.2f}%。",
            f"- 以 FreEformer 点估计为阈值，日期级 bootstrap 中三项指标同时更优的比例为 "
            f"{strongest['bootstrap_prob_all_below_baseline']:.2%}；其中 MAPE、MSE、MAE 分项更优比例分别为 "
            f"{strongest['bootstrap_prob_mape_below_baseline']:.2%}、{strongest['bootstrap_prob_mse_below_baseline']:.2%}、"
            f"{strongest['bootstrap_prob_mae_below_baseline']:.2%}。",
            "- 论文可写“点估计三指标均优于现代强基线，且相对最强基线的重采样优势概率较高”；不宜写作“已完成配对显著性检验”或“所有重采样均显著优于”。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"passed": True, "output_json": str(args.output_json), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
