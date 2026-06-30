"""Evaluate a fixed ablation checkpoint suite with the current data protocol.

The script is evidence-oriented: it does not select checkpoints on the test
set. Each checkpoint is assumed to be a validation-saved ``best_model_latest``
artifact, then validation and test metrics are recomputed once with the
train-only scaler from ``prepare_data``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from data_loader import prepare_data
from evaluate_validation_ensemble import (
    collect_predictions,
    compute_metrics,
    load_checkpoint_model,
)


DEFAULT_SUITE = {
    "RAMR-full(scene+regime+robust+entropy)": "../checkpoints/paper_experiments/ramr_full_robust/best_model_latest.pth",
    "w/o scene gating": "../checkpoints/paper_experiments/ablation_no_scene/best_model_latest.pth",
    "w/o regime routing": "../checkpoints/paper_experiments/ablation_no_regime/best_model_latest.pth",
    "variance load balance": "../checkpoints/paper_experiments/ablation_variance_balance/best_model_latest.pth",
    "distribution basic(mean/std/max)": "../checkpoints/strict_no_leakage/stat_basic/best_model_latest.pth",
    "distribution quantile": "../checkpoints/strict_no_leakage/stat_quantile/best_model_latest.pth",
    "MAE-aware robust": "../checkpoints/strict_no_leakage/mae_robust/best_model_latest.pth",
}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_extra(items: list[str] | None) -> dict[str, str]:
    parsed = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--checkpoint must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed


def _config_view(config: dict) -> dict:
    keys = [
        "ablation_mode",
        "use_scene_gating",
        "use_regime_routing",
        "statistic_feature_set",
        "enhanced_statistic",
        "balance_mode",
        "lambda_balance",
        "lambda_mape",
        "lambda_mae",
        "lambda_mse",
        "lambda_range",
        "lookback",
        "horizon",
        "hidden_dim",
        "num_experts",
        "scene_dim",
        "regime_dim",
    ]
    return {key: config.get(key) for key in keys if key in config}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed ablation checkpoints")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="../results/fixed_ablation_suite_20260701")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint",
        action="append",
        help="Optional additional checkpoint in NAME=PATH form.",
    )
    parser.add_argument("--no_default_suite", action="store_true")
    args = parser.parse_args()

    suite = {} if args.no_default_suite else dict(DEFAULT_SUITE)
    suite.update(_parse_extra(args.checkpoint))
    if not suite:
        raise ValueError("No checkpoints to evaluate.")

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_suite = {name: _resolve(path) for name, path in suite.items()}
    missing = {name: str(path) for name, path in resolved_suite.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(json.dumps(missing, ensure_ascii=False, indent=2))

    first = torch.load(next(iter(resolved_suite.values())), map_location="cpu", weights_only=False)
    config = first["config"]
    train_loader, val_loader, test_loader, scaler, dataset_info = prepare_data(
        data_dir=args.data_dir,
        target_cols=None,
        lookback=config["lookback"],
        horizon=config["horizon"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_both=True,
    )
    del train_loader

    n_targets = dataset_info["n_targets"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []

    for name, checkpoint_path in resolved_suite.items():
        checkpoint_cpu = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ck_config = checkpoint_cpu["config"]
        if ck_config["lookback"] != config["lookback"] or ck_config["horizon"] != config["horizon"]:
            raise ValueError(f"Incompatible lookback/horizon for {name}: {checkpoint_path}")

        model, checkpoint = load_checkpoint_model(checkpoint_path, device)
        val_preds, val_targets = collect_predictions(model, val_loader, device, scaler, n_targets)
        test_preds, test_targets = collect_predictions(model, test_loader, device, scaler, n_targets)
        val_metrics = compute_metrics(val_preds, val_targets)
        test_metrics = compute_metrics(test_preds, test_targets)

        row = {
            "name": name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "checkpoint_best_val_mape": checkpoint.get("best_val_mape"),
            "val_mape": val_metrics["mape"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "test_mape": test_metrics["mape"],
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "valid_count": test_metrics["valid_count"],
            "ignored_count": test_metrics["ignored_count"],
        }
        row.update({f"config_{k}": v for k, v in _config_view(ck_config).items()})
        rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "fixed_ablation_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "protocol": (
            "Fixed checkpoint evaluation only. Checkpoints are validation-saved artifacts; "
            "test metrics are recomputed once with the train-only scaler."
        ),
        "data_dir": str(_resolve(args.data_dir)),
        "output_dir": str(output_dir),
        "n_targets": int(n_targets),
        "target_cols": dataset_info.get("target_cols"),
        "rows": df.to_dict(orient="records"),
    }
    with open(output_dir / "fixed_ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
