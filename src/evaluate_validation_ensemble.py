"""
Validation-only checkpoint ensemble for MoE-Nanjin.

This script prevents test leakage by using the validation set for every
selection decision: candidate filtering, ensemble weights, and optional global
scale calibration. Candidate test metrics are never computed or exported; the
test set is evaluated only after the final recipe is fixed from validation
metrics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import prepare_data
from model import MoENanjin
from utils import mape_with_threshold


def _strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_checkpoint_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    ablation_mode = config.get("ablation_mode")
    if ablation_mode is None:
        ablation_mode = "baseline"

    model = MoENanjin(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        lookback=config["lookback"],
        num_experts=config["num_experts"],
        hidden_dim=config["hidden_dim"],
        dropout=0.1,
        ablation_mode=ablation_mode,
        use_scene_gating=config.get("use_scene_gating", False),
        enhanced_statistic=config.get("enhanced_statistic", True),
        statistic_feature_set=config.get("statistic_feature_set", "robust"),
        use_regime_routing=config.get("use_regime_routing", False),
        regime_dim=config.get("regime_dim", 16),
        scene_dim=config.get("scene_dim", 0),
    ).to(device)
    model.load_state_dict(_strip_module_prefix(checkpoint["model_state_dict"]))
    model.eval()
    return model, checkpoint


def collect_predictions(model, data_loader, device, scaler, n_targets):
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_scene = batch
                batch_scene = batch_scene.to(device)
            else:
                batch_x, batch_y = batch
                batch_scene = None
            batch_x = batch_x.to(device)
            output, _, _, _ = model(batch_x, batch_scene)
            all_preds.append(output.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).reshape(-1, n_targets)
    targets = np.concatenate(all_targets, axis=0).reshape(-1, n_targets)
    preds_orig = np.clip(scaler.inverse_transform(preds), 0, None)
    targets_orig = scaler.inverse_transform(targets)
    return preds_orig, targets_orig


def compute_metrics(preds, targets):
    preds = np.clip(preds, 0, None)
    mape, valid_count, ignored_count = mape_with_threshold(
        targets.flatten(), preds.flatten(), threshold=10.0
    )
    mse = np.sum((targets - preds) ** 2) / (np.sum(targets ** 2) + 1e-8)
    mae = np.sum(np.abs(targets - preds)) / (np.sum(targets) + 1e-8)
    return {
        "mape": float(mape),
        "mse": float(mse),
        "mae": float(mae),
        "valid_count": int(valid_count),
        "ignored_count": int(ignored_count),
    }


def objective(metrics, baseline):
    return (
        metrics["mape"] / baseline["mape"]
        + metrics["mse"] / baseline["mse"]
        + metrics["mae"] / baseline["mae"]
    ) / 3.0


def apply_weights(pred_stack, weights, scale=1.0):
    return np.tensordot(weights, pred_stack, axes=(0, 0)) * scale


def optimize_scale(preds, targets, metric_name, baseline):
    candidates = np.linspace(0.90, 1.10, 401)
    best = None
    for scale in candidates:
        metrics = compute_metrics(preds * scale, targets)
        score = metrics[metric_name] if metric_name != "balanced" else objective(metrics, baseline)
        if best is None or score < best["score"]:
            best = {"scale": float(scale), "metrics": metrics, "score": float(score)}
    return best


def candidate_weight_sets(n_models, rng, random_samples):
    weights = []
    for i in range(n_models):
        one = np.zeros(n_models, dtype=np.float64)
        one[i] = 1.0
        weights.append(one)
    weights.append(np.ones(n_models, dtype=np.float64) / n_models)
    for _ in range(random_samples):
        weights.append(rng.dirichlet(np.ones(n_models)))
    return weights


def score_weight_batch(weights, pred_stack_flat, target_flat, metric_name, baseline):
    preds = weights @ pred_stack_flat
    preds = np.clip(preds, 0, None)
    target = target_flat.reshape(1, -1)

    valid_mask = (target > 10.0) & (preds > 10.0)
    valid_counts = valid_mask.sum(axis=1)
    ape = np.abs((target - preds) / (target + 1e-8))
    mape_sum = (ape * valid_mask).sum(axis=1)
    mape = np.divide(
        mape_sum * 100.0,
        valid_counts,
        out=np.full(weights.shape[0], np.inf, dtype=np.float64),
        where=valid_counts > 0,
    )

    sq_error = (target - preds) ** 2
    mse = sq_error.sum(axis=1) / (np.sum(target_flat ** 2) + 1e-8)
    mae = np.abs(target - preds).sum(axis=1) / (np.sum(target_flat) + 1e-8)

    if metric_name == "mape":
        score = mape
    elif metric_name == "mse":
        score = mse
    elif metric_name == "mae":
        score = mae
    else:
        score = (
            mape / baseline["mape"] +
            mse / baseline["mse"] +
            mae / baseline["mae"]
        ) / 3.0
    return score, mape, mse, mae


def select_ensemble(val_stack, val_targets, metric_name, random_samples, seed, baseline, calibrate_top_k):
    rng = np.random.default_rng(seed)
    n_models = val_stack.shape[0]
    pred_stack_flat = val_stack.reshape(n_models, -1).astype(np.float64)
    target_flat = val_targets.reshape(-1).astype(np.float64)
    ranked = []

    deterministic = np.stack(candidate_weight_sets(n_models, rng, 0), axis=0)
    batches = [deterministic]
    random_left = random_samples
    batch_size = 2048
    while random_left > 0:
        current = min(batch_size, random_left)
        batches.append(rng.dirichlet(np.ones(n_models), size=current))
        random_left -= current

    for weights_batch in batches:
        score, mape, mse, mae = score_weight_batch(
            weights_batch, pred_stack_flat, target_flat, metric_name, baseline
        )
        top_idx = np.argsort(score)[:calibrate_top_k]
        for idx in top_idx:
            ranked.append((
                float(score[idx]),
                weights_batch[idx].copy(),
                {
                    "mape": float(mape[idx]),
                    "mse": float(mse[idx]),
                    "mae": float(mae[idx]),
                },
            ))
        ranked.sort(key=lambda item: item[0])
        ranked = ranked[:calibrate_top_k]

    best = None
    for _, weights, _ in ranked[:calibrate_top_k]:
        preds = apply_weights(val_stack, weights)
        scale_result = optimize_scale(preds, val_targets, metric_name, baseline)
        metrics = scale_result["metrics"]
        score = metrics[metric_name] if metric_name != "balanced" else objective(metrics, baseline)
        if best is None or score < best["score"]:
            best = {
                "weights": weights,
                "scale": scale_result["scale"],
                "val_metrics": metrics,
                "score": float(score),
            }
    return best


def parse_checkpoint_args(paths):
    checkpoints = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            checkpoints.extend(sorted(path.glob("best_model_epoch_*.pth")))
        else:
            checkpoints.append(path)
    unique = []
    seen = set()
    for path in checkpoints:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Validation-only ensemble evaluator")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output_dir", type=str, default="../results/validation_ensemble_correct_data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--random_samples", type=int, default=20000)
    parser.add_argument("--calibrate_top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection_metric", choices=["balanced", "mape", "mse", "mae"], default="balanced")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoints = parse_checkpoint_args(args.checkpoint)
    if not checkpoints:
        raise ValueError("No checkpoints were provided.")

    first_checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    config = first_checkpoint["config"]
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

    val_preds = []
    rows = []
    val_targets_ref = None

    for checkpoint_path in checkpoints:
        model, checkpoint = load_checkpoint_model(checkpoint_path, device)
        ckpt_config = checkpoint["config"]
        if ckpt_config["lookback"] != config["lookback"] or ckpt_config["horizon"] != config["horizon"]:
            raise ValueError(f"Incompatible checkpoint shape: {checkpoint_path}")

        val_pred, val_targets = collect_predictions(model, val_loader, device, scaler, n_targets)
        if val_targets_ref is None:
            val_targets_ref = val_targets
        else:
            if not np.allclose(val_targets_ref, val_targets):
                raise ValueError(f"Target mismatch for checkpoint: {checkpoint_path}")

        val_metric = compute_metrics(val_pred, val_targets_ref)
        val_preds.append(val_pred)
        rows.append({
            "name": checkpoint_path.parent.name + "/" + checkpoint_path.name,
            "path": str(checkpoint_path),
            "val_mape": val_metric["mape"],
            "val_mse": val_metric["mse"],
            "val_mae": val_metric["mae"],
            "checkpoint_val_mape": checkpoint.get("val_mape"),
        })
        del model

    candidate_df = pd.DataFrame(rows)
    candidate_df.to_csv(output_dir / "candidate_metrics.csv", index=False, encoding="utf-8-sig")

    val_stack = np.stack(val_preds, axis=0)
    baseline = {"mape": 19.41, "mse": 0.037700, "mae": 0.144000}

    best = select_ensemble(
        val_stack,
        val_targets_ref,
        metric_name=args.selection_metric,
        random_samples=args.random_samples,
        seed=args.seed,
        baseline=baseline,
        calibrate_top_k=args.calibrate_top_k,
    )

    final_val_pred = apply_weights(val_stack, best["weights"], best["scale"])
    final_val_metrics = compute_metrics(final_val_pred, val_targets_ref)

    weights_df = pd.DataFrame({
        "name": candidate_df["name"],
        "path": candidate_df["path"],
        "weight": best["weights"],
    })
    weights_df = weights_df[weights_df["weight"] > 1e-4].sort_values("weight", ascending=False)
    weights_df.to_csv(output_dir / "selected_weights.csv", index=False, encoding="utf-8-sig")

    test_preds = []
    test_targets_ref = None
    selected_weights = []
    for checkpoint_path, weight in zip(checkpoints, best["weights"]):
        if weight <= 0:
            continue
        model, _ = load_checkpoint_model(checkpoint_path, device)
        test_pred, test_targets = collect_predictions(model, test_loader, device, scaler, n_targets)
        if test_targets_ref is None:
            test_targets_ref = test_targets
        elif not np.allclose(test_targets_ref, test_targets):
            raise ValueError(f"Test target mismatch for checkpoint: {checkpoint_path}")
        test_preds.append(test_pred)
        selected_weights.append(weight)
        del model

    test_stack = np.stack(test_preds, axis=0)
    selected_weights = np.asarray(selected_weights, dtype=np.float64)
    selected_weights = selected_weights / selected_weights.sum()
    final_test_pred = apply_weights(test_stack, selected_weights, best["scale"])
    final_test_metrics = compute_metrics(final_test_pred, test_targets_ref)

    summary = {
        "protocol": "Weights and scale are selected on validation set only; test set is evaluated once after selection.",
        "selection_metric": args.selection_metric,
        "random_samples": args.random_samples,
        "calibrate_top_k": args.calibrate_top_k,
        "seed": args.seed,
        "baseline_threshold": baseline,
        "scale_selected_on_validation": best["scale"],
        "validation_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "beats_baseline": {
            "mape": final_test_metrics["mape"] < baseline["mape"],
            "mse": final_test_metrics["mse"] < baseline["mse"],
            "mae": final_test_metrics["mae"] < baseline["mae"],
            "all": (
                final_test_metrics["mape"] < baseline["mape"]
                and final_test_metrics["mse"] < baseline["mse"]
                and final_test_metrics["mae"] < baseline["mae"]
            ),
        },
        "selected_weights": weights_df.to_dict(orient="records"),
    }
    with open(output_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
