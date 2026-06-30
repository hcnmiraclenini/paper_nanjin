#!/usr/bin/env python3
"""Paired audit against modern baselines with exported test predictions.

This audit uses a fresh modern-baseline rerun that saves validation-selected
test predictions for each baseline. It compares those predictions with the
fixed Strict RAMR-VE final-test predictions on the same test dates and target
variables. No model training, tuning, checkpoint selection, or recalibration is
performed by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAMR = ROOT / "docs/experiments/artifacts/strict_ramr_ve_test_predictions_20260701.npz"
DEFAULT_BASELINE_DIR = ROOT / "docs/experiments/artifacts/modern_baseline_paired_predictions_20260701"
DEFAULT_JSON = ROOT / "docs/experiments/artifacts/modern_baseline_paired_audit_20260701.json"
DEFAULT_CSV = ROOT / "docs/experiments/artifacts/modern_baseline_paired_audit_20260701.csv"
DEFAULT_MD = ROOT / "docs/experiments/现代强基线配对预测审计_20260701.md"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
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


def bootstrap_pair(
    ramr_preds: np.ndarray,
    base_preds: np.ndarray,
    targets: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n_rows = targets.shape[0]
    diffs = {"mape": [], "mse": [], "mae": []}
    wins = {"mape": 0, "mse": 0, "mae": 0, "all": 0}
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        ramr_m = metrics(ramr_preds[idx], targets[idx])
        base_m = metrics(base_preds[idx], targets[idx])
        all_win = True
        for metric in ["mape", "mse", "mae"]:
            diff = float(base_m[metric] - ramr_m[metric])
            diffs[metric].append(diff)
            if diff > 0:
                wins[metric] += 1
            else:
                all_win = False
        if all_win:
            wins["all"] += 1

    out = {}
    for metric, values in diffs.items():
        arr = np.asarray(values, dtype=np.float64)
        out[metric] = {
            "mean_baseline_minus_ramr": float(arr.mean()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "prob_ramr_better": float(wins[metric] / n_boot),
        }
    out["prob_all_three_ramr_better"] = float(wins["all"] / n_boot)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramr_predictions", type=Path, default=DEFAULT_RAMR)
    parser.add_argument("--baseline_prediction_dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ramr_payload = np.load(args.ramr_predictions)
    ramr_preds = ramr_payload["preds"]
    targets = ramr_payload["targets"]
    ramr_point = metrics(ramr_preds, targets)

    files = sorted(args.baseline_prediction_dir.glob("*_test_predictions.npz"))
    if not files:
        raise FileNotFoundError(f"No baseline prediction files found under {args.baseline_prediction_dir}")

    rows = []
    details = []
    for path in files:
        model = path.name.replace("_test_predictions.npz", "")
        payload = np.load(path)
        base_preds = payload["preds"]
        base_targets = payload["targets"]
        if base_preds.shape != ramr_preds.shape:
            raise ValueError(f"Prediction shape mismatch for {model}: {base_preds.shape} vs {ramr_preds.shape}")
        if not np.allclose(base_targets, targets):
            raise ValueError(f"Target mismatch for {model}; paired audit requires identical test targets.")
        base_point = metrics(base_preds, targets)
        boot = bootstrap_pair(ramr_preds, base_preds, targets, args.bootstrap_samples, args.seed)
        row = {
            "baseline": model,
            "baseline_mape": base_point["mape"],
            "baseline_mse": base_point["mse"],
            "baseline_mae": base_point["mae"],
            "ramr_mape": ramr_point["mape"],
            "ramr_mse": ramr_point["mse"],
            "ramr_mae": ramr_point["mae"],
            "mape_margin": base_point["mape"] - ramr_point["mape"],
            "mse_margin": base_point["mse"] - ramr_point["mse"],
            "mae_margin": base_point["mae"] - ramr_point["mae"],
            "mape_prob_ramr_better": boot["mape"]["prob_ramr_better"],
            "mse_prob_ramr_better": boot["mse"]["prob_ramr_better"],
            "mae_prob_ramr_better": boot["mae"]["prob_ramr_better"],
            "all_three_prob_ramr_better": boot["prob_all_three_ramr_better"],
        }
        rows.append(row)
        details.append({"baseline": model, "prediction_file": portable_path(path), "point": base_point, "bootstrap": boot})

    df = pd.DataFrame(rows).sort_values("all_three_prob_ramr_better")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    result = {
        "passed": bool(
            len(rows) == 5
            and (df[["mape_margin", "mse_margin", "mae_margin"]] > 0).all().all()
            and (df["all_three_prob_ramr_better"] >= 0.70).all()
        ),
        "protocol": (
            "Paired post-hoc audit only. Baseline predictions are loaded from a validation-selected "
            "modern-baseline rerun with exported predictions; this script performs no training, "
            "checkpoint selection, tuning, or calibration."
        ),
        "ramr_prediction_file": portable_path(args.ramr_predictions),
        "baseline_prediction_dir": portable_path(args.baseline_prediction_dir),
        "n_rows": int(targets.shape[0]),
        "n_targets": int(targets.shape[1]),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "ramr_point": ramr_point,
        "rows": rows,
        "details": details,
        "summary": {
            "point_dominance_cells": int((df[["mape_margin", "mse_margin", "mae_margin"]] > 0).sum().sum()),
            "point_dominance_total_cells": int(len(df) * 3),
            "min_all_three_prob": float(df["all_three_prob_ramr_better"].min()),
            "max_all_three_prob": float(df["all_three_prob_ramr_better"].max()),
            "closest_by_all_three_prob": str(df.iloc[0]["baseline"]),
        },
    }
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 现代强基线配对预测审计（2026-07-01）",
        "",
        "## 审计口径",
        "",
        "- 本审计读取已导出的 Strict RAMR-VE 固定测试预测和现代强基线 prediction-export rerun 测试预测。",
        "- 所有模型使用相同测试日期和相同20个目标变量；脚本逐文件校验 `targets` 完全一致。",
        "- 本脚本不训练、不调参、不重新选模、不做尺度校准，仅做配对误差统计和日期级 bootstrap。",
        "- 该审计与旧现代强基线点估计表分开记录；旧表用于主文基线对比，本审计用于补充逐样本可比性证据。",
        "",
        "## 配对点估计与 bootstrap 胜率",
        "",
        "| 基线 | 基线MAPE/% | 基线MSE | 基线MAE | RAMR MAPE/% | RAMR MSE | RAMR MAE | 三指标同时更优概率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.to_dict(orient="records"):
        lines.append(
            f"| {row['baseline']} | {row['baseline_mape']:.4f} | {row['baseline_mse']:.6f} | "
            f"{row['baseline_mae']:.6f} | {row['ramr_mape']:.4f} | {row['ramr_mse']:.6f} | "
            f"{row['ramr_mae']:.6f} | {row['all_three_prob_ramr_better']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- 点估计支配单元：{result['summary']['point_dominance_cells']}/{result['summary']['point_dominance_total_cells']}。",
            f"- 三指标同时更优的日期级 bootstrap 概率范围为 {result['summary']['min_all_three_prob']:.2%} 至 {result['summary']['max_all_three_prob']:.2%}。",
            "- 由于本审计使用的是带预测导出的现代基线重跑结果，主文应写作补充配对证据，不替换原现代强基线主表。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output_json": str(args.output_json),
                "point_dominance_cells": result["summary"]["point_dominance_cells"],
                "min_all_three_prob": result["summary"]["min_all_three_prob"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
