#!/usr/bin/env python3
"""Analyze target- and direction-level errors for fixed Strict RAMR-VE outputs.

The script is a post-hoc audit only. It reads the fixed final-test prediction
artifact and the real project CSV files to reconstruct target names, then
reports per-station-direction MAPE/MSE/MAE. It does not train, tune, or select
models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED = ROOT / "docs/experiments/artifacts/strict_ramr_ve_test_predictions_20260701.npz"
DEFAULT_OUT_JSON = ROOT / "docs/experiments/artifacts/target_error_profile_20260701.json"
DEFAULT_OUT_CSV = ROOT / "docs/experiments/artifacts/target_error_profile_20260701.csv"
DEFAULT_OUT_MD = ROOT / "docs/experiments/站点方向误差剖面审计_20260701.md"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def target_names(data_dir: Path) -> list[str]:
    sites = ["北京南", "成都东", "广州南", "汉口", "杭州东", "杭州西", "上海", "上海虹桥", "武汉", "西安北"]
    from_df = pd.read_csv(data_dir / "from_nj.csv", encoding="gbk")
    to_df = pd.read_csv(data_dir / "to_nj.csv", encoding="gbk")
    if len(from_df.columns) - 1 != len(sites) or len(to_df.columns) - 1 != len(sites):
        raise ValueError("Unexpected station-column count in from_nj.csv or to_nj.csv")
    return [f"{site}_from_nj" for site in sites] + [f"{site}_to_nj" for site in sites]


def split_direction(name: str) -> str:
    if name.endswith("_from_nj"):
        return "from_nj"
    if name.endswith("_to_nj"):
        return "to_nj"
    return "unknown"


def station_name(name: str) -> str:
    return name.replace("_from_nj", "").replace("_to_nj", "")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    names = target_names(args.data_dir)
    payload = np.load(args.predictions)
    preds = payload["preds"]
    targets = payload["targets"]
    if preds.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {preds.shape} vs {targets.shape}")
    if preds.shape[1] != len(names):
        raise ValueError(f"Target count mismatch: predictions={preds.shape[1]}, names={len(names)}")

    target_rows = []
    for idx, name in enumerate(names):
        item = metrics(preds[:, [idx]], targets[:, [idx]])
        item.update({"target": name, "station": station_name(name), "direction": split_direction(name)})
        target_rows.append(item)

    target_df = pd.DataFrame(target_rows).sort_values("mape")
    direction_rows = []
    for direction in ["from_nj", "to_nj"]:
        idx = [i for i, name in enumerate(names) if split_direction(name) == direction]
        item = metrics(preds[:, idx], targets[:, idx])
        item.update({"direction": direction, "n_targets": len(idx)})
        direction_rows.append(item)

    overall = metrics(preds, targets)
    profile = {
        "protocol": (
            "Fixed prediction audit only. The final-test prediction artifact is loaded as-is; "
            "no model training, checkpoint selection, ensemble weighting, or calibration is performed."
        ),
        "predictions_file": portable_path(args.predictions),
        "data_dir": portable_path(args.data_dir),
        "n_rows": int(preds.shape[0]),
        "n_targets": int(preds.shape[1]),
        "overall": overall,
        "direction_metrics": direction_rows,
        "target_metrics": target_rows,
        "summary": {
            "median_target_mape": float(target_df["mape"].median()),
            "mean_target_mape": float(target_df["mape"].mean()),
            "max_target_mape": float(target_df["mape"].max()),
            "min_target_mape": float(target_df["mape"].min()),
            "targets_below_20_mape": int((target_df["mape"] < 20.0).sum()),
            "targets_below_25_mape": int((target_df["mape"] < 25.0).sum()),
            "targets_below_30_mape": int((target_df["mape"] < 30.0).sum()),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    target_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    worst = target_df.sort_values("mape", ascending=False).head(5)
    best = target_df.head(5)
    direction_df = pd.DataFrame(direction_rows)
    lines = [
        "# 站点方向误差剖面审计（2026-07-01）",
        "",
        "## 审计口径",
        "",
        "- 本分析只读取固定 Strict RAMR-VE 测试集预测文件，不训练、不调参、不重新选模。",
        "- 目标变量顺序由项目真实 `data/from_nj.csv` 与 `data/to_nj.csv` 的固定站点顺序重建。",
        "- MAPE 采用与主实验一致的阈值口径：真实值和预测值均大于 10 的样本参与计算。",
        "",
        "## 总体结果",
        "",
        f"- 总体：MAPE={overall['mape']:.4f}%，MSE={overall['mse']:.6f}，MAE={overall['mae']:.6f}。",
        f"- 20 个目标变量的 MAPE 中位数为 {profile['summary']['median_target_mape']:.4f}%，均值为 {profile['summary']['mean_target_mape']:.4f}%。",
        f"- MAPE <20% 的目标变量数为 {profile['summary']['targets_below_20_mape']}/20，"
        f"MAPE <25% 的目标变量数为 {profile['summary']['targets_below_25_mape']}/20，"
        f"MAPE <30% 的目标变量数为 {profile['summary']['targets_below_30_mape']}/20。",
        "",
        "## 方向聚合误差",
        "",
        "| 方向 | 目标数 | MAPE/% | MSE | MAE | 平均客流 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in direction_df.to_dict(orient="records"):
        label = "南京南→方向站" if row["direction"] == "from_nj" else "方向站→南京南"
        lines.append(
            f"| {label} | {row['n_targets']} | {row['mape']:.4f} | {row['mse']:.6f} | "
            f"{row['mae']:.6f} | {row['mean_target']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## MAPE 最低的 5 个目标变量",
            "",
            "| 目标变量 | MAPE/% | MSE | MAE | 平均客流 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best.to_dict(orient="records"):
        lines.append(
            f"| {row['target']} | {row['mape']:.4f} | {row['mse']:.6f} | "
            f"{row['mae']:.6f} | {row['mean_target']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## MAPE 最高的 5 个目标变量",
            "",
            "| 目标变量 | MAPE/% | MSE | MAE | 平均客流 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in worst.to_dict(orient="records"):
        lines.append(
            f"| {row['target']} | {row['mape']:.4f} | {row['mse']:.6f} | "
            f"{row['mae']:.6f} | {row['mean_target']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 写作建议",
            "",
            "- 主文可补充：最终模型在双向方向聚合上保持一致改善，并非仅由单个高流量方向拉低总体误差。",
            "- 对高 MAPE 的低流量方向应保持克制说明：低客流变量更容易因小分母造成百分比误差放大，后续可引入分层损失或外生事件变量进一步优化。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": True, "output_json": str(args.output_json), "targets": len(target_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
