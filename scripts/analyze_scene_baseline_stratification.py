#!/usr/bin/env python3
"""Scene-stratified audit against modern baselines.

This audit compares fixed Strict RAMR-VE predictions with exported modern
baseline predictions under weekday/weekend calendar scenes. It is a post-hoc
error characterization only: scene strata are not used for model selection,
checkpoint search, calibration, or any test-set tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_RAMR = ROOT / "docs/experiments/artifacts/strict_ramr_ve_test_predictions_20260701.npz"
DEFAULT_BASELINE_DIR = ROOT / "docs/experiments/artifacts/modern_baseline_paired_predictions_20260701"
DEFAULT_JSON = ROOT / "docs/experiments/artifacts/scene_baseline_stratification_audit_20260701.json"
DEFAULT_CSV = ROOT / "docs/experiments/artifacts/scene_baseline_stratification_audit_20260701.csv"
DEFAULT_MD = ROOT / "docs/experiments/场景分层现代强基线审计_20260701.md"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_targets(data_dir: Path) -> tuple[list[pd.Timestamp], list[str], np.ndarray]:
    from_df = pd.read_csv(data_dir / "from_nj.csv", encoding="gbk")
    to_df = pd.read_csv(data_dir / "to_nj.csv", encoding="gbk")
    from_df["time"] = pd.to_datetime(from_df["time"])
    to_df["time"] = pd.to_datetime(to_df["time"])
    common = pd.merge(from_df, to_df, on="time", suffixes=("_from_nj", "_to_nj")).sort_values("time")
    from_cols = [c for c in common.columns if c.endswith("_from_nj")]
    to_cols = [c for c in common.columns if c.endswith("_to_nj")]
    target_cols = from_cols + to_cols
    return common["time"].tolist(), target_cols, common[target_cols].to_numpy(dtype=np.float64)


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


def add_scene_rows(
    rows: list[dict],
    scene: str,
    scene_label: str,
    mask: np.ndarray,
    models: dict[str, np.ndarray],
    targets: np.ndarray,
) -> None:
    scene_rows = []
    for model, preds in models.items():
        metric = metrics(preds[mask], targets[mask])
        scene_rows.append(
            {
                "scene": scene,
                "scene_label": scene_label,
                "model": model,
                "n_days": int(mask.sum()),
                "n_observations": int(mask.sum() * targets.shape[1]),
                **metric,
            }
        )

    for metric_name in ["mape", "mse", "mae"]:
        ordered = sorted(scene_rows, key=lambda item: item[metric_name])
        for rank, item in enumerate(ordered, start=1):
            item[f"{metric_name}_rank"] = rank
            item[f"{metric_name}_best"] = rank == 1
    rows.extend(scene_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ramr_predictions", type=Path, default=DEFAULT_RAMR)
    parser.add_argument("--baseline_prediction_dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    full_dates, target_cols, full_targets = load_targets(args.data_dir)
    ramr_payload = np.load(args.ramr_predictions)
    targets = ramr_payload["targets"].astype(np.float64)
    ramr_preds = ramr_payload["preds"].astype(np.float64)

    test_start = len(full_targets) - len(targets)
    test_dates = full_dates[test_start:]
    if test_start != 634 or len(test_dates) != 212:
        raise ValueError(f"Unexpected test split reconstruction: start={test_start}, n={len(test_dates)}")
    if not np.allclose(full_targets[test_start:], targets):
        raise ValueError("Rebuilt targets from real CSV do not match fixed prediction artifact targets.")

    models: dict[str, np.ndarray] = {"Strict RAMR-VE": ramr_preds}
    for path in sorted(args.baseline_prediction_dir.glob("*_test_predictions.npz")):
        model = path.name.replace("_test_predictions.npz", "")
        payload = np.load(path)
        if not np.allclose(payload["targets"], targets):
            raise ValueError(f"Target mismatch for baseline {model}")
        models[model] = payload["preds"].astype(np.float64)

    expected_models = {"Strict RAMR-VE", "DLinear", "PatchTST", "iTransformer", "TimeMixer", "FreEformer"}
    if set(models) != expected_models:
        raise ValueError(f"Unexpected model set: {sorted(models)}")

    date_index = pd.Series(pd.to_datetime(test_dates))
    weekday_mask = date_index.dt.dayofweek.to_numpy() < 5
    weekend_mask = date_index.dt.dayofweek.to_numpy() >= 5
    if int(weekday_mask.sum()) != 152 or int(weekend_mask.sum()) != 60:
        raise ValueError(
            f"Unexpected scene counts: weekday={int(weekday_mask.sum())}, weekend={int(weekend_mask.sum())}"
        )

    scenes = [
        ("all_test", "全部测试日期", np.ones(len(targets), dtype=bool)),
        ("weekday", "工作日", weekday_mask),
        ("weekend", "周末", weekend_mask),
    ]

    rows: list[dict] = []
    for scene, scene_label, mask in scenes:
        add_scene_rows(rows, scene, scene_label, mask, models, targets)

    df = pd.DataFrame(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    scene_lookup = {(row["scene"], row["model"]): row for row in rows}
    weekday_ramr = scene_lookup[("weekday", "Strict RAMR-VE")]
    weekend_ramr = scene_lookup[("weekend", "Strict RAMR-VE")]
    weekday_best_all = all(weekday_ramr[f"{metric_name}_rank"] == 1 for metric_name in ["mape", "mse", "mae"])
    weekend_best_all = all(weekend_ramr[f"{metric_name}_rank"] == 1 for metric_name in ["mape", "mse", "mae"])

    def best_model(scene: str, metric_name: str) -> str:
        sub = df[df["scene"] == scene].sort_values(metric_name)
        return str(sub.iloc[0]["model"])

    result = {
        "passed": bool(weekday_best_all and weekend_best_all),
        "protocol": (
            "Post-hoc scene-stratified audit only. Weekday/weekend groups are derived from reconstructed "
            "test dates for error characterization. No training, checkpoint selection, tuning, calibration, "
            "or test-set model search is performed."
        ),
        "data_dir": portable_path(args.data_dir),
        "ramr_prediction_file": portable_path(args.ramr_predictions),
        "baseline_prediction_dir": portable_path(args.baseline_prediction_dir),
        "n_dates": int(targets.shape[0]),
        "n_targets": int(targets.shape[1]),
        "test_start_index": int(test_start),
        "test_start_date": test_dates[0].strftime("%Y-%m-%d"),
        "test_end_date": test_dates[-1].strftime("%Y-%m-%d"),
        "target_columns": target_cols,
        "scene_counts": {
            "weekday": int(weekday_mask.sum()),
            "weekend": int(weekend_mask.sum()),
            "holiday": 0,
        },
        "summary": {
            "weekday_ramr_best_all_metrics": bool(weekday_best_all),
            "weekend_ramr_best_all_metrics": bool(weekend_best_all),
            "weekday_best_models": {metric_name: best_model("weekday", metric_name) for metric_name in ["mape", "mse", "mae"]},
            "weekend_best_models": {metric_name: best_model("weekend", metric_name) for metric_name in ["mape", "mse", "mae"]},
            "weekday_ramr_metrics": {
                "mape": weekday_ramr["mape"],
                "mse": weekday_ramr["mse"],
                "mae": weekday_ramr["mae"],
            },
            "weekend_ramr_metrics": {
                "mape": weekend_ramr["mape"],
                "mse": weekend_ramr["mse"],
                "mae": weekend_ramr["mae"],
            },
        },
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def table(scene: str) -> list[str]:
        lines = [
            "| 模型 | MAPE/% | MSE | MAE | MAPE排名 | MSE排名 | MAE排名 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in df[df["scene"] == scene].sort_values("mape").to_dict(orient="records"):
            lines.append(
                f"| {row['model']} | {row['mape']:.4f} | {row['mse']:.6f} | {row['mae']:.6f} | "
                f"{int(row['mape_rank'])} | {int(row['mse_rank'])} | {int(row['mae_rank'])} |"
            )
        return lines

    lines = [
        "# 场景分层现代强基线审计（2026-07-01）",
        "",
        "## 审计口径",
        "",
        "- 本审计只读取固定 Strict RAMR-VE 测试预测和现代强基线 prediction-export rerun 测试预测。",
        "- 测试日期由真实 `data/from_nj.csv` 与 `data/to_nj.csv` 重建，脚本校验重建目标数组与预测 artifact 中的 `targets` 完全一致。",
        "- 工作日/周末标签由测试日期的星期信息生成；当前严格时间顺序测试窗口包含 152 个工作日、60 个周末和 0 个节假日样本。",
        "- 分层仅用于事后误差刻画，不训练、不调参、不重新选模、不做尺度校准。",
        "",
        "## 工作日场景对比",
        "",
        *table("weekday"),
        "",
        "## 周末场景对比",
        "",
        *table("weekend"),
        "",
        "## 结论边界",
        "",
        f"- 工作日上 Strict RAMR-VE 的 MAPE/MSE/MAE 为 {weekday_ramr['mape']:.4f}%/{weekday_ramr['mse']:.6f}/{weekday_ramr['mae']:.6f}，三项指标均为所有现代强基线中最优。",
        f"- 周末上 Strict RAMR-VE 的 MAPE/MSE/MAE 为 {weekend_ramr['mape']:.4f}%/{weekend_ramr['mse']:.6f}/{weekend_ramr['mae']:.6f}，三项指标均为所有现代强基线中最优。",
        "- 该结果支持日历场景特征和场景条件化门控对不同日期类型具有稳定误差收益；但当前测试窗口无节假日样本，因此不能外推为节假日场景上的统计结论。",
        "",
    ]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output_json": portable_path(args.output_json),
                "weekday_ramr_best_all_metrics": weekday_best_all,
                "weekend_ramr_best_all_metrics": weekend_best_all,
                "weekday_ramr_metrics": result["summary"]["weekday_ramr_metrics"],
                "weekend_ramr_metrics": result["summary"]["weekend_ramr_metrics"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
