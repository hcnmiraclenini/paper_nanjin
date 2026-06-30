"""Export predictions and bootstrap confidence intervals for a fixed ensemble.

This script does not select checkpoints, weights, or calibration scales. It
loads a previously fixed validation-selected recipe from ``selected_weights``
and ``ensemble_summary`` and evaluates the validation/test loaders once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import prepare_data
from evaluate_validation_ensemble import (
    apply_weights,
    collect_predictions,
    compute_metrics,
    load_checkpoint_model,
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _bootstrap_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    n_boot: int,
    seed: int,
    baseline: dict | None = None,
) -> tuple[dict, dict | None]:
    rng = np.random.default_rng(seed)
    n_rows = targets.shape[0]
    samples = {"mape": [], "mse": [], "mae": []}
    baseline_wins = {key: 0 for key in samples}
    baseline_all_wins = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        metrics = compute_metrics(preds[idx], targets[idx])
        wins_all = True
        for key in samples:
            samples[key].append(metrics[key])
            if baseline and metrics[key] < baseline[key]:
                baseline_wins[key] += 1
            elif baseline:
                wins_all = False
        if baseline and wins_all:
            baseline_all_wins += 1

    ci = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=np.float64)
        ci[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    if not baseline:
        return ci, None
    win_rate = {key: float(value / n_boot) for key, value in baseline_wins.items()}
    win_rate["all_metrics"] = float(baseline_all_wins / n_boot)
    return ci, win_rate


def _prediction_frame(preds: np.ndarray, targets: np.ndarray, target_cols: list[str]) -> pd.DataFrame:
    rows = {}
    for idx, col in enumerate(target_cols):
        rows[f"{col}_true"] = targets[:, idx]
        rows[f"{col}_pred"] = preds[:, idx]
        rows[f"{col}_abs_error"] = np.abs(targets[:, idx] - preds[:, idx])
    return pd.DataFrame(rows)


def _load_recipe(weights_path: Path, summary_path: Path) -> tuple[pd.DataFrame, float, dict]:
    weights_df = pd.read_csv(weights_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    scale = float(summary["scale_selected_on_validation"])
    required = {"name", "path", "weight"}
    missing = required.difference(weights_df.columns)
    if missing:
        raise ValueError(f"selected_weights missing columns: {sorted(missing)}")
    if weights_df.empty:
        raise ValueError("selected_weights is empty")
    weights_df = weights_df[weights_df["weight"] > 0].copy()
    weights_df["weight"] = weights_df["weight"].astype(float)
    weights_df["path"] = weights_df["path"].map(lambda p: str(_resolve(p)))
    weight_sum = float(weights_df["weight"].sum())
    if weight_sum <= 0:
        raise ValueError("selected weights sum to zero")
    weights_df["weight"] = weights_df["weight"] / weight_sum
    return weights_df, scale, summary


def _collect_fixed_ensemble(
    weights_df: pd.DataFrame,
    loader,
    device: torch.device,
    scaler,
    n_targets: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_list = []
    target_ref = None
    for row in weights_df.itertuples(index=False):
        checkpoint_path = Path(row.path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        model, _ = load_checkpoint_model(checkpoint_path, device)
        preds, targets = collect_predictions(model, loader, device, scaler, n_targets)
        if target_ref is None:
            target_ref = targets
        elif not np.allclose(target_ref, targets):
            raise ValueError(f"Target mismatch for checkpoint: {checkpoint_path}")
        pred_list.append(preds)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stack = np.stack(pred_list, axis=0)
    weights = weights_df["weight"].to_numpy(dtype=np.float64)
    return stack, target_ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed validation-selected ensemble predictions")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights_path = _resolve(args.weights)
    summary_path = _resolve(args.summary)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_df, scale, summary = _load_recipe(weights_path, summary_path)
    baseline = summary.get("baseline_threshold")
    first_checkpoint = torch.load(weights_df.iloc[0]["path"], map_location="cpu", weights_only=False)
    config = first_checkpoint["config"]
    _, val_loader, test_loader, scaler, dataset_info = prepare_data(
        data_dir=args.data_dir,
        target_cols=None,
        lookback=config["lookback"],
        horizon=config["horizon"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_both=True,
    )
    n_targets = dataset_info["n_targets"]
    target_cols = dataset_info.get("target_cols", [f"target_{i}" for i in range(n_targets)])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    outputs = {}
    for split, loader in [("validation", val_loader), ("test", test_loader)]:
        stack, targets = _collect_fixed_ensemble(weights_df, loader, device, scaler, n_targets)
        weights = weights_df["weight"].to_numpy(dtype=np.float64)
        preds = apply_weights(stack, weights, scale)
        metrics = compute_metrics(preds, targets)
        ci, baseline_win_rate = _bootstrap_metrics(
            preds, targets, args.bootstrap, args.seed, baseline=baseline
        )
        frame = _prediction_frame(preds, targets, target_cols)
        frame.to_csv(output_dir / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(output_dir / f"{split}_predictions.npz", preds=preds, targets=targets)
        outputs[split] = {
            "metrics": metrics,
            "bootstrap_ci": ci,
            "bootstrap_probability_below_baseline": baseline_win_rate,
            "n_rows": int(targets.shape[0]),
            "n_targets": int(targets.shape[1]),
        }

    result = {
        "protocol": "Fixed recipe export only: checkpoints, weights, and scale are loaded from validation-selected files.",
        "weights_file": str(weights_path),
        "summary_file": str(summary_path),
        "scale_selected_on_validation": scale,
        "selection_metric": summary.get("selection_metric"),
        "bootstrap_unit": "test/validation date row with all target variables kept together",
        "bootstrap_samples": args.bootstrap,
        "seed": args.seed,
        "validation": outputs["validation"],
        "test": outputs["test"],
    }
    with open(output_dir / "fixed_ensemble_prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
