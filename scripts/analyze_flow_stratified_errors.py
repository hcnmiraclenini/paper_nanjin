#!/usr/bin/env python3
"""Flow-stratified error audit for fixed Strict RAMR-VE predictions.

This is a post-hoc robustness audit. It reads the fixed final-test prediction
artifact and evaluates errors by passenger-flow magnitude and by daily total
demand. It does not train, tune, select checkpoints, or recalibrate outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED = ROOT / "docs/experiments/artifacts/strict_ramr_ve_test_predictions_20260701.npz"
DEFAULT_OUT_JSON = ROOT / "docs/experiments/artifacts/flow_stratified_error_audit_20260701.json"
DEFAULT_OUT_CSV = ROOT / "docs/experiments/artifacts/flow_stratified_error_audit_20260701.csv"
DEFAULT_OUT_MD = ROOT / "docs/experiments/流量分层误差审计_20260701.md"


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
        "mean_target": float(np.mean(targets)),
        "total_target": float(np.sum(targets)),
        "total_abs_error": float(np.sum(np.abs(targets - preds))),
    }


def flat_row(label: str, group_type: str, preds: np.ndarray, targets: np.ndarray, extra: dict | None = None) -> dict:
    row = {"group_type": group_type, "group": label, "n_observations": int(targets.size)}
    row.update(metrics(preds.reshape(-1, 1), targets.reshape(-1, 1)))
    if extra:
        row.update(extra)
    return row


def day_row(label: str, preds: np.ndarray, targets: np.ndarray, extra: dict | None = None) -> dict:
    row = {"group_type": "daily_total", "group": label, "n_days": int(targets.shape[0]), "n_observations": int(targets.size)}
    row.update(metrics(preds, targets))
    if extra:
        row.update(extra)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    payload = np.load(args.predictions)
    preds = np.clip(payload["preds"], 0, None)
    targets = payload["targets"]
    if preds.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {preds.shape} vs {targets.shape}")

    rows: list[dict] = []
    flat_pred = preds.reshape(-1)
    flat_target = targets.reshape(-1)

    absolute_bins = [
        ("low_flow_<500", flat_target < 500, {"threshold": "<500"}),
        ("mid_flow_500_3000", (flat_target >= 500) & (flat_target < 3000), {"threshold": "500-3000"}),
        ("high_flow_>=3000", flat_target >= 3000, {"threshold": ">=3000"}),
    ]
    for label, mask, extra in absolute_bins:
        rows.append(flat_row(label, "flow_magnitude", flat_pred[mask], flat_target[mask], extra))

    q33, q66 = np.quantile(flat_target, [0.33, 0.66])
    quantile_bins = [
        ("target_low_tertile", flat_target <= q33, {"lower": None, "upper": float(q33)}),
        ("target_mid_tertile", (flat_target > q33) & (flat_target <= q66), {"lower": float(q33), "upper": float(q66)}),
        ("target_high_tertile", flat_target > q66, {"lower": float(q66), "upper": None}),
    ]
    for label, mask, extra in quantile_bins:
        rows.append(flat_row(label, "target_tertile", flat_pred[mask], flat_target[mask], extra))

    daily_total = targets.sum(axis=1)
    q25, q75 = np.quantile(daily_total, [0.25, 0.75])
    day_bins = [
        ("low_demand_days", daily_total <= q25, {"upper_daily_total": float(q25)}),
        ("normal_demand_days", (daily_total > q25) & (daily_total < q75), {"lower_daily_total": float(q25), "upper_daily_total": float(q75)}),
        ("high_demand_days", daily_total >= q75, {"lower_daily_total": float(q75)}),
    ]
    for label, mask, extra in day_bins:
        rows.append(day_row(label, preds[mask], targets[mask], extra))

    overall = metrics(preds, targets)
    df = pd.DataFrame(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": (
            "Fixed final-test prediction audit only. Flow strata are derived from observed test targets "
            "for post-hoc error characterization, not for model selection or recalibration."
        ),
        "predictions_file": portable_path(args.predictions),
        "n_rows": int(preds.shape[0]),
        "n_targets": int(preds.shape[1]),
        "overall": overall,
        "flow_quantiles": {"q33": float(q33), "q66": float(q66)},
        "daily_total_quantiles": {"q25": float(q25), "q75": float(q75)},
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    lookup = {row["group"]: row for row in rows}
    low_abs = lookup["low_flow_<500"]
    mid_abs = lookup["mid_flow_500_3000"]
    high_abs = lookup["high_flow_>=3000"]
    low_day = lookup["low_demand_days"]
    normal_day = lookup["normal_demand_days"]
    high_day = lookup["high_demand_days"]

    lines = [
        "# 流量分层误差审计（2026-07-01）",
        "",
        "## 审计口径",
        "",
        "- 本分析只读取固定 Strict RAMR-VE 测试集预测文件，不训练、不调参、不重新选模。",
        "- 分层只用于事后误差刻画，不用于模型选择或尺度校准。",
        "- 单点客流规模分层按真实客流值划分：低流量 <500，中等流量 500-3000，高流量 >=3000。",
        "- 日期需求分层按测试集中每日 20 个目标变量总客流的 25% 与 75% 分位数划分。",
        "",
        "## 单点客流规模分层",
        "",
        "| 分层 | 样本数 | 平均客流 | MAPE/% | MSE | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [low_abs, mid_abs, high_abs]:
        label = {"low_flow_<500": "<500", "mid_flow_500_3000": "500-3000", "high_flow_>=3000": ">=3000"}[row["group"]]
        lines.append(
            f"| {label} | {row['n_observations']} | {row['mean_target']:.2f} | "
            f"{row['mape']:.4f} | {row['mse']:.6f} | {row['mae']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 日期总需求分层",
            "",
            f"- 日期总需求 25% 分位数为 {q25:.2f}，75% 分位数为 {q75:.2f}。",
            "",
            "| 分层 | 日期数 | 平均日总客流 | MAPE/% | MSE | MAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in [low_day, normal_day, high_day]:
        label = {
            "low_demand_days": "低需求日",
            "normal_demand_days": "常规需求日",
            "high_demand_days": "高需求日",
        }[row["group"]]
        lines.append(
            f"| {label} | {row['n_days']} | {row['total_target'] / row['n_days']:.2f} | "
            f"{row['mape']:.4f} | {row['mse']:.6f} | {row['mae']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- 高单点客流样本（>=3000）的 MAPE 为 {high_abs['mape']:.4f}%，低于低流量样本（<500）的 {low_abs['mape']:.4f}%，说明主业务高流量样本的相对误差更稳定。",
            f"- 高需求日 MAPE 为 {high_day['mape']:.4f}%，高于常规需求日的 {normal_day['mape']:.4f}%，说明极端或集中需求日期仍是后续优化重点。",
            "- 论文应将该结果写作误差分层与局限分析，而不是新的选模依据。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": True, "output_json": str(args.output_json), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
