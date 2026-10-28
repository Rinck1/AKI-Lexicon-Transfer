#!/usr/bin/env python3
"""
09 — Deep learning baseline (MLP / tabular-feature Transformer)

HA-AKI, first stay, onset+48h features → hospital mortality.

Protocol:
  - Data: data/processed/stay_features_48h.parquet
  - Split: existing subject-level train/val/test column
  - Train-only median impute + z-score
  - Early stop on validation AUROC
  - Outputs: outputs/tables/dl_{mlp|transformer}_performance.csv

Usage:
  python scripts/09_dl_baseline.py
  python scripts/09_dl_baseline.py --model mlp
  python scripts/09_dl_baseline.py --model transformer
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# Project root = parent of scripts/
PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# ---------------------------------------------------------------------------
# Defaults (anti-overfit)
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 256
N_EPOCHS = 100
PATIENCE = 15

MLP_HIDDEN = (64, 32)
MLP_DROPOUT = 0.45
MLP_LR = 5e-4
MLP_WEIGHT_DECAY = 5e-3

TF_HIDDEN = 64
TF_HEADS = 4
TF_LAYERS = 1
TF_DROPOUT = 0.4
TF_LR = 3e-4
TF_WEIGHT_DECAY = 1e-2

GRAD_CLIP = 1.0

FEATURE_COLS_V1 = [
    "AGE",
    "GENDER_M",
    "onset_stage",
    "onset_hour_from_icu",
    "CHF",
    "Diabetes",
    "HTN",
    "CKD_FLAG",
    "COPD",
    "LiverDisease",
    "Arrhythmia",
    "urine_output_ml_mean",
    "urine_output_ml_last",
    "urine_output_ml_min",
    "urine_output_ml_miss_frac",
    "gcs_total_mean",
    "gcs_total_last",
    "gcs_total_miss_frac",
    "gcs_motor_last",
    "gcs_eye_last",
    "sbp_mean",
    "sbp_last",
    "sbp_miss_frac",
    "dbp_mean",
    "mbp_mean",
    "heart_rate_mean",
    "heart_rate_last",
    "respiratory_rate_mean",
    "respiratory_rate_last",
    "spo2_mean",
    "spo2_last",
    "spo2_miss_frac",
    "creatinine_mean",
    "creatinine_last",
    "creatinine_max",
    "creatinine_miss_frac",
    "bun_mean",
    "bun_last",
    "bicarbonate_mean",
    "bicarbonate_last",
    "potassium_mean",
    "sodium_mean",
    "lactate_mean",
    "lactate_last",
    "lactate_miss_frac",
    "anion_gap_mean",
    "rdw_mean",
    "rdw_last",
    "wbc_mean",
    "platelet_mean",
    "hemoglobin_mean",
    "glucose_mean",
]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StayFeatureDataset(Dataset):
    """One row per ICU stay; tabular features only."""

    def __init__(self, x: pd.DataFrame, y: pd.Series):
        self.x = torch.as_tensor(x.to_numpy(dtype=np.float32), dtype=torch.float32)
        self.y = torch.as_tensor(y.to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class MortalityMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] = MLP_HIDDEN,
        dropout: float = MLP_DROPOUT,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TabularTransformer(nn.Module):
    """Each scalar feature is one token (tabular), NOT 48h time series."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = TF_HIDDEN,
        num_heads: int = TF_HEADS,
        num_layers: int = TF_LAYERS,
        dropout: float = TF_DROPOUT,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(1, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_proj(x.unsqueeze(-1))
        h = h + self.pos_embedding
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


def prepare_splits(df: pd.DataFrame, feature_cols: list[str]):
    x_full = df[feature_cols]
    y_full = df["HOSPITAL_MORTALITY"]
    train_m = df["split"] == "train"
    val_m = df["split"] == "val"
    test_m = df["split"] == "test"

    x_train = x_full.loc[train_m].copy()
    x_val = x_full.loc[val_m].copy()
    x_test = x_full.loc[test_m].copy()
    y_train = y_full.loc[train_m]
    y_val = y_full.loc[val_m]
    y_test = y_full.loc[test_m]

    fill_values = x_train.median(numeric_only=True).fillna(0.0)
    x_train = x_train.fillna(fill_values)
    x_val = x_val.fillna(fill_values)
    x_test = x_test.fillna(fill_values)

    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    x_train = (x_train - means) / stds
    x_val = (x_val - means) / stds
    x_test = (x_test - means) / stds

    return x_train, y_train, x_val, y_val, x_test, y_test


@torch.no_grad()
def collect_probs(net: nn.Module, loader: DataLoader, device: torch.device):
    net.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = net(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="09 DL baseline (MLP / Transformer)")
    parser.add_argument(
        "--model",
        choices=["mlp", "transformer"],
        default="mlp",
        help="Model type (default: mlp)",
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    model_type = args.model.lower()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("09: Deep learning baseline (MLP / tabular Transformer)")
    print("=" * 60)
    print(f"  device={device}  model={model_type}  seed={args.seed}")

    data_path = PROJ / "data" / "processed" / "stay_features_48h.parquet"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Missing {data_path}. Run pipeline through 06 first."
        )
    df = pd.read_parquet(data_path)
    missing = [c for c in FEATURE_COLS_V1 if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    print(f"  features={len(FEATURE_COLS_V1)}  rows={len(df)}")

    x_train, y_train, x_val, y_val, x_test, y_test = prepare_splits(df, FEATURE_COLS_V1)
    print(
        f"  split train/val/test={len(x_train)}/{len(x_val)}/{len(x_test)} | "
        f"train mortality={float(y_train.mean()):.3f}"
    )

    train_loader = DataLoader(
        StayFeatureDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        StayFeatureDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        StayFeatureDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    input_dim = x_train.shape[1]
    if model_type == "transformer":
        net = TabularTransformer(input_dim).to(device)
        lr, wd = TF_LR, TF_WEIGHT_DECAY
        print(
            f"  TabularTransformer H={TF_HIDDEN} L={TF_LAYERS} "
            f"dropout={TF_DROPOUT} lr={lr} wd={wd}"
        )
    else:
        net = MortalityMLP(input_dim).to(device)
        lr, wd = MLP_LR, MLP_WEIGHT_DECAY
        print(
            f"  MortalityMLP hidden={MLP_HIDDEN} dropout={MLP_DROPOUT} "
            f"lr={lr} wd={wd}"
        )

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  trainable params={n_params:,}")

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
    )
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1.0)], device=device, dtype=torch.float32
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  pos_weight={pos_weight.item():.3f}")

    best_val_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        net.train()
        running, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(xb), yb)
            loss.backward()
            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running / max(n_seen, 1)

        net.eval()
        v_running, v_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_running += criterion(net(xb), yb).item() * xb.size(0)
                v_n += xb.size(0)
        val_loss = v_running / max(v_n, 1)

        tr_prob, tr_y = collect_probs(net, train_loader, device)
        va_prob, va_y = collect_probs(net, val_loader, device)
        train_auc = roc_auc_score(tr_y, tr_prob)
        val_auc = roc_auc_score(va_y, va_prob)
        gap = train_auc - val_auc
        scheduler.step(val_auc)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss tr/va {train_loss:.4f}/{val_loss:.4f} | "
            f"AUC tr/va {train_auc:.4f}/{val_auc:.4f} | gap={gap:.4f}"
        )

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, best val AUC={best_val_auc:.4f})"
                )
                break

    if best_state is None:
        raise RuntimeError("No checkpoint saved.")
    net.load_state_dict(best_state)
    net.to(device)
    print(f"Loaded best checkpoint epoch={best_epoch} val_AUC={best_val_auc:.4f}")

    def evaluate(loader: DataLoader, name: str) -> dict:
        y_prob, y_true = collect_probs(net, loader, device)
        m = metrics_from_probs(y_true, y_prob)
        print(f"\n{'=' * 50}")
        print(f"  {name}  (n={m['n']})")
        print(f"  AUC:   {m['auc']:.4f}")
        print(f"  AUPRC: {m['auprc']:.4f}")
        print(f"  Brier: {m['brier']:.4f}")
        print(f"{'=' * 50}")
        return {"split": name, **m}

    rows = [
        evaluate(train_loader, "Train"),
        evaluate(val_loader, "Val"),
        evaluate(test_loader, "Test"),
    ]
    print(
        f"\nSummary | train={rows[0]['auc']:.4f} val={rows[1]['auc']:.4f} "
        f"test={rows[2]['auc']:.4f} | gap(train-test)="
        f"{rows[0]['auc'] - rows[2]['auc']:.4f}"
    )

    out_dir = PROJ / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dl_{model_type}_performance.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("09 done.")


if __name__ == "__main__":
    main()
