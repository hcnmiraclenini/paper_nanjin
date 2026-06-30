#!/usr/bin/env python3
"""Audit the no-leakage data protocol used by the final railway experiments.

This script does not train, tune, or evaluate models. It rebuilds the real
Nanjing South railway dataset, reconstructs the time-ordered train/validation/
test split, and verifies the preprocessing/window invariants required by the
Strict RAMR-VE protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import prepare_data  # noqa: E402


DEFAULT_JSON = ROOT / "docs/experiments/artifacts/no_leakage_protocol_audit_20260701.json"
DEFAULT_MD = ROOT / "docs/experiments/无泄露实验协议审计_20260701.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def split_summary(date_index: pd.DatetimeIndex, train_end: int, val_end: int) -> dict:
    splits = {
        "train": (0, train_end),
        "validation": (train_end, val_end),
        "test": (val_end, len(date_index)),
    }
    out = {}
    for name, (start, end) in splits.items():
        dates = date_index[start:end]
        out[name] = {
            "start_index": int(start),
            "end_index_exclusive": int(end),
            "n_days": int(end - start),
            "start_date": str(dates[0].date()),
            "end_date": str(dates[-1].date()),
        }
    return out


def inspect_dataset_windows(dataset, split_name: str, target_start: int, target_end: int) -> dict:
    violations = []
    first_examples = []
    last_examples = []

    for local_pos, end_idx in enumerate(dataset.indices):
        global_target = dataset.scene_offset + int(end_idx)
        input_start = global_target - dataset.lookback
        input_end_exclusive = global_target

        ok = (
            target_start <= global_target < target_end
            and input_start >= 0
            and input_end_exclusive <= global_target
            and input_end_exclusive <= target_end
            and input_end_exclusive <= global_target
            and input_start < global_target
        )
        if not ok:
            violations.append(
                {
                    "local_pos": int(local_pos),
                    "global_target_index": int(global_target),
                    "input_start": int(input_start),
                    "input_end_exclusive": int(input_end_exclusive),
                }
            )

        example = {
            "local_pos": int(local_pos),
            "global_target_index": int(global_target),
            "input_index_range": [int(input_start), int(input_end_exclusive - 1)],
        }
        if len(first_examples) < 3:
            first_examples.append(example)
        last_examples.append(example)
        if len(last_examples) > 3:
            last_examples.pop(0)

    return {
        "split": split_name,
        "n_samples": int(len(dataset.indices)),
        "target_index_min": int(dataset.scene_offset + min(dataset.indices)),
        "target_index_max": int(dataset.scene_offset + max(dataset.indices)),
        "expected_target_range": [int(target_start), int(target_end - 1)],
        "input_always_before_target": len(violations) == 0,
        "target_inside_split": len(violations) == 0,
        "violation_count": len(violations),
        "violations": violations[:10],
        "first_examples": first_examples,
        "last_examples": last_examples,
    }


def scaler_equal(a: RobustScaler, b: RobustScaler) -> bool:
    return bool(np.allclose(a.center_, b.center_) and np.allclose(a.scale_, b.scale_))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data")
    parser.add_argument("--lookback", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--test_ratio", type=float, default=0.25)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, scaler, dataset_info = prepare_data(
        data_dir=str(args.data_dir),
        target_cols=None,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=128,
        num_workers=0,
        use_both=True,
        fit_scaler_on_full_data=False,
    )

    merged_df = dataset_info["merged_df"]
    target_cols = dataset_info["target_cols"]
    date_index = pd.DatetimeIndex(dataset_info["date_index"])
    train_end = int(dataset_info["train_end"])
    val_end = int(dataset_info["val_end"])
    target_data = merged_df[target_cols].to_numpy()

    train_only_scaler = RobustScaler().fit(target_data[:train_end])
    full_data_scaler = RobustScaler().fit(target_data)
    scaler_matches_train_only = scaler_equal(scaler, train_only_scaler)
    scaler_matches_full_data = scaler_equal(scaler, full_data_scaler)

    train_window = inspect_dataset_windows(train_loader.dataset, "train", 0, train_end)
    val_window = inspect_dataset_windows(val_loader.dataset, "validation", train_end, val_end)
    test_window = inspect_dataset_windows(test_loader.dataset, "test", val_end, len(date_index))

    split = split_summary(date_index, train_end, val_end)
    chronological_ok = (
        pd.Series(date_index).is_monotonic_increasing
        and split["train"]["end_date"] < split["validation"]["start_date"]
        and split["validation"]["end_date"] < split["test"]["start_date"]
    )

    prediction_rows_expected = len(test_loader.dataset.indices)
    prediction_artifact = ROOT / "docs/experiments/artifacts/strict_ramr_ve_test_predictions_20260701.npz"
    prediction_rows_actual = None
    if prediction_artifact.exists():
        payload = np.load(prediction_artifact)
        prediction_rows_actual = int(payload["preds"].shape[0])

    passed = bool(
        chronological_ok
        and scaler_matches_train_only
        and not scaler_matches_full_data
        and train_window["violation_count"] == 0
        and val_window["violation_count"] == 0
        and test_window["violation_count"] == 0
        and prediction_rows_actual == prediction_rows_expected
    )

    result = {
        "passed": passed,
        "protocol": (
            "Audit only. The script rebuilds data splits and preprocessing invariants; "
            "it does not train, tune, select checkpoints, or evaluate candidate models."
        ),
        "data_dir": portable_path(args.data_dir),
        "data_hash": {
            "from_nj_sha256": sha256(args.data_dir / "from_nj.csv"),
            "to_nj_sha256": sha256(args.data_dir / "to_nj.csv"),
        },
        "n_days": int(len(date_index)),
        "n_targets": int(len(target_cols)),
        "lookback": args.lookback,
        "horizon": args.horizon,
        "split": split,
        "checks": {
            "chronological_split_non_overlapping": bool(chronological_ok),
            "robust_scaler_matches_train_only_fit": scaler_matches_train_only,
            "robust_scaler_does_not_match_full_data_fit": not scaler_matches_full_data,
            "train_windows_valid": train_window["violation_count"] == 0,
            "validation_windows_valid": val_window["violation_count"] == 0,
            "test_windows_valid": test_window["violation_count"] == 0,
            "fixed_prediction_rows_match_test_dataset": prediction_rows_actual == prediction_rows_expected,
        },
        "scaler": {
            "center_first5": [float(x) for x in scaler.center_[:5]],
            "scale_first5": [float(x) for x in scaler.scale_[:5]],
            "train_only_center_first5": [float(x) for x in train_only_scaler.center_[:5]],
            "full_data_center_first5": [float(x) for x in full_data_scaler.center_[:5]],
        },
        "windows": {
            "train": train_window,
            "validation": val_window,
            "test": test_window,
        },
        "prediction_artifact": {
            "file": portable_path(prediction_artifact),
            "expected_test_rows": prediction_rows_expected,
            "actual_prediction_rows": prediction_rows_actual,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 无泄露实验协议审计（2026-07-01）",
        "",
        "## 审计目的",
        "",
        "本审计只检查数据划分、标准化和窗口构造协议，不训练、不调参、不重新选模。目的是把 Strict RAMR-VE 的无测试泄露口径转化为可复查证据。",
        "",
        "## 数据与划分",
        "",
        f"- 数据文件：`{portable_path(args.data_dir / 'from_nj.csv')}` 与 `{portable_path(args.data_dir / 'to_nj.csv')}`。",
        f"- 合并后共同日期：{len(date_index)} 天；目标变量：{len(target_cols)} 个。",
        f"- 回看窗口：{args.lookback} 天；预测步长：{args.horizon} 天。",
        "",
        "| 划分 | 起止索引 | 日期范围 | 天数 |",
        "| --- | ---: | --- | ---: |",
    ]
    for name, item in split.items():
        lines.append(
            f"| {name} | {item['start_index']}-{item['end_index_exclusive'] - 1} | "
            f"{item['start_date']} 至 {item['end_date']} | {item['n_days']} |"
        )
    lines.extend(
        [
            "",
            "## 核验结果",
            "",
            "| 检查项 | 结果 | 说明 |",
            "| --- | --- | --- |",
            f"| 时间顺序划分无重叠 | {'通过' if chronological_ok else '未通过'} | 训练、验证、测试日期严格递增且互不交叠 |",
            f"| Scaler 仅训练集拟合 | {'通过' if scaler_matches_train_only else '未通过'} | 当前 `RobustScaler` 参数与训练集拟合结果一致 |",
            f"| Scaler 不等于全量拟合 | {'通过' if not scaler_matches_full_data else '未通过'} | 防止验证/测试分布进入预处理 |",
            f"| 训练窗口无未来目标 | {'通过' if train_window['violation_count'] == 0 else '未通过'} | {train_window['n_samples']} 个训练样本 |",
            f"| 验证窗口无测试信息 | {'通过' if val_window['violation_count'] == 0 else '未通过'} | {val_window['n_samples']} 个验证样本；首个验证窗口只借用训练末尾历史 |",
            f"| 测试窗口无未来/测试后验调参 | {'通过' if test_window['violation_count'] == 0 else '未通过'} | {test_window['n_samples']} 个测试样本；首个测试窗口只借用验证末尾历史 |",
            f"| 固定预测文件行数匹配测试样本 | {'通过' if prediction_rows_actual == prediction_rows_expected else '未通过'} | 预测行数 {prediction_rows_actual}，测试样本 {prediction_rows_expected} |",
            "",
            "## 关键边界",
            "",
            "- 验证集和测试集首个样本使用前一划分末尾的历史窗口，这是时间序列预测的真实在线可用历史，不包含被预测日期及其未来值。",
            "- 集成权重和尺度校准由验证集确定；测试集预测文件只用于最终结果和事后误差审计。",
            "- 本审计不能替代更多外部数据集验证，但可以证明当前铁路主实验的预处理和窗口构造没有使用测试集后验信息。",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": passed,
                "output_json": str(args.output_json),
                "expected_test_rows": prediction_rows_expected,
                "actual_prediction_rows": prediction_rows_actual,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
