"""
Modern baseline experiments for the Nanjin railway passenger-flow dataset.

The implementations are compact reproductions adapted to the local daily
multivariate dataset. They keep the same data split and metrics as MoE-Rail.
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import prepare_data
from utils import mape_with_threshold

plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei", "SimHei", "Noto Sans CJK SC",
    "Microsoft YaHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DLinearBaseline(nn.Module):
    def __init__(self, lookback, n_targets, horizon):
        super().__init__()
        self.linear = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(n_targets)])
        self.n_targets = n_targets
        self.horizon = horizon

    def forward(self, x):
        outs = [self.linear[i](x[:, :, i]) for i in range(self.n_targets)]
        return torch.stack(outs, dim=-1).reshape(x.size(0), self.horizon * self.n_targets)


class PatchTSTBaseline(nn.Module):
    def __init__(self, lookback, n_targets, horizon, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_targets = n_targets
        self.horizon = horizon
        self.patch_len = 2
        self.stride = 1
        n_patches = max(1, (lookback - self.patch_len) // self.stride + 1)
        self.patch_proj = nn.Linear(self.patch_len * n_targets, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon * n_targets))

    def forward(self, x):
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().reshape(x.size(0), patches.size(1), -1)
        z = self.patch_proj(patches) + self.pos[:, :patches.size(1), :]
        z = self.encoder(z).mean(dim=1)
        return self.head(z)


class ITransformerBaseline(nn.Module):
    def __init__(self, lookback, n_targets, horizon, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_targets = n_targets
        self.horizon = horizon
        self.var_proj = nn.Linear(lookback, d_model)
        self.var_embed = nn.Parameter(torch.randn(1, n_targets, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x):
        z = self.var_proj(x.transpose(1, 2)) + self.var_embed
        z = self.encoder(z)
        y = self.head(z).transpose(1, 2)
        return y.reshape(x.size(0), self.horizon * self.n_targets)


class TimeMixerBaseline(nn.Module):
    def __init__(self, lookback, n_targets, horizon, hidden=256, dropout=0.1):
        super().__init__()
        self.n_targets = n_targets
        self.horizon = horizon
        in_dim = lookback * n_targets + 2 * n_targets + n_targets
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * n_targets),
        )

    def forward(self, x):
        raw = x.reshape(x.size(0), -1)
        scale2 = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True).mean(dim=-1)
        scale3 = F.avg_pool1d(x.transpose(1, 2), kernel_size=3, stride=3, ceil_mode=True).mean(dim=-1)
        last_delta = x[:, -1, :] - x[:, 0, :]
        return self.net(torch.cat([raw, scale2, scale3, last_delta], dim=-1))


class FreEformerBaseline(nn.Module):
    def __init__(self, lookback, n_targets, horizon, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_targets = n_targets
        self.horizon = horizon
        n_freq = lookback // 2 + 1
        self.freq_proj = nn.Linear(n_freq * 2, d_model)
        self.var_embed = nn.Parameter(torch.randn(1, n_targets, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x):
        freq = torch.fft.rfft(x.transpose(1, 2), dim=-1)
        feats = torch.cat([freq.real, freq.imag], dim=-1)
        z = self.freq_proj(feats) + self.var_embed
        z = self.encoder(z)
        y = self.head(z).transpose(1, 2)
        return y.reshape(x.size(0), self.horizon * self.n_targets)


def calculate_metrics(model, loader, device, scaler, n_targets, horizon):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            bx, by = batch[0].to(device), batch[1].to(device)
            out = model(bx)
            preds.append(out.cpu().numpy())
            targets.append(by.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    preds_orig = scaler.inverse_transform(preds.reshape(-1, n_targets))
    targets_orig = scaler.inverse_transform(targets.reshape(-1, n_targets))
    preds_orig = np.clip(preds_orig, 0, None)
    mape, valid_count, ignored_count = mape_with_threshold(targets_orig.flatten(), preds_orig.flatten(), threshold=10.0)
    mse = np.sum((targets_orig - preds_orig) ** 2) / (np.sum(targets_orig ** 2) + 1e-8)
    mae = np.sum(np.abs(targets_orig - preds_orig)) / (np.sum(targets_orig) + 1e-8)
    return {
        "mape": float(mape),
        "mse": float(mse),
        "mae": float(mae),
        "valid_count": int(valid_count),
        "ignored_count": int(ignored_count),
    }


def train_one(name, model, train_loader, val_loader, test_loader, scaler, n_targets, horizon, args, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    best = {"val_mape": float("inf"), "epoch": 0, "state": None}
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            bx, by = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            out = model(bx)
            loss = F.mse_loss(out, by) + 0.2 * F.l1_loss(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        val = calculate_metrics(model, val_loader, device, scaler, n_targets, horizon)
        sched.step(val["mape"])
        if val["mape"] < best["val_mape"]:
            best = {
                "val_mape": val["mape"],
                "epoch": epoch,
                "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"{name} epoch={epoch:03d} train={np.mean(losses):.4f} val_mape={val['mape']:.2f}%")
        if patience >= args.patience:
            break

    model.load_state_dict(best["state"])
    test = calculate_metrics(model, test_loader, device, scaler, n_targets, horizon)
    return {
        "model": name,
        "best_epoch": int(best["epoch"]),
        "val_mape": float(best["val_mape"]),
        "test_mape": test["mape"],
        "test_mse": test["mse"],
        "test_mae": test["mae"],
    }


def plot_results(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = df.sort_values("test_mape")
    ax.bar(order["model"], order["test_mape"], color="#4C72B0")
    ax.set_ylabel("测试集MAPE/%")
    ax.set_title("现代时序基线在铁路客流数据集上的对比")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(order["test_mape"]):
        ax.text(i, v + 0.2, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "modern_baselines_mape.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="../results/modern_baselines")
    parser.add_argument("--lookback", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, scaler, info = prepare_data(
        data_dir=args.data_dir,
        target_cols=None,
        lookback=args.lookback,
        horizon=args.horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_both=True,
    )
    n_targets = info["n_targets"]
    models = {
        "DLinear": DLinearBaseline(args.lookback, n_targets, args.horizon),
        "PatchTST": PatchTSTBaseline(args.lookback, n_targets, args.horizon, d_model=args.hidden_dim),
        "iTransformer": ITransformerBaseline(args.lookback, n_targets, args.horizon, d_model=args.hidden_dim),
        "TimeMixer": TimeMixerBaseline(args.lookback, n_targets, args.horizon, hidden=args.hidden_dim * 2),
        "FreEformer": FreEformerBaseline(args.lookback, n_targets, args.horizon, d_model=args.hidden_dim),
    }

    results = []
    for name, model in models.items():
        print(f"\\n===== {name} =====")
        result = train_one(name, model, train_loader, val_loader, test_loader, scaler, n_targets, args.horizon, args, device)
        print(result)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "modern_baselines_results.csv", index=False)
    with open(out_dir / "modern_baselines_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plot_results(df, out_dir)
    print(df.to_string(index=False))
    print(f"[OK] saved to {out_dir}")


if __name__ == "__main__":
    main()
