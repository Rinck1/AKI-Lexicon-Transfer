#!/usr/bin/env python3
"""
10 — Compare HistGBM / MLP / tabular Transformer with a richer metric suite.

Same data protocol as 08/09:
  stay_features_48h, subject splits, FEATURE_COLS_V1 for DL,
  all numeric features for HistGBM (as in 08).

Outputs:
  outputs/tables/baseline_models_rich_metrics.csv
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from scripts.utils.constants import RANDOM_SEED  # noqa: E402

SEED = RANDOM_SEED
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same feature set as 09 for fair DL comparison
FEATURE_COLS_V1 = [
    "AGE", "GENDER_M", "onset_stage", "onset_hour_from_icu",
    "CHF", "Diabetes", "HTN", "CKD_FLAG", "COPD", "LiverDisease", "Arrhythmia",
    "urine_output_ml_mean", "urine_output_ml_last", "urine_output_ml_min", "urine_output_ml_miss_frac",
    "gcs_total_mean", "gcs_total_last", "gcs_total_miss_frac",
    "gcs_motor_last", "gcs_eye_last",
    "sbp_mean", "sbp_last", "sbp_miss_frac", "dbp_mean", "mbp_mean",
    "heart_rate_mean", "heart_rate_last",
    "respiratory_rate_mean", "respiratory_rate_last",
    "spo2_mean", "spo2_last", "spo2_miss_frac",
    "creatinine_mean", "creatinine_last", "creatinine_max", "creatinine_miss_frac",
    "bun_mean", "bun_last", "bicarbonate_mean", "bicarbonate_last",
    "potassium_mean", "sodium_mean",
    "lactate_mean", "lactate_last", "lactate_miss_frac",
    "anion_gap_mean", "rdw_mean", "rdw_last",
    "wbc_mean", "platelet_mean", "hemoglobin_mean", "glucose_mean",
]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def rich_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thr: float | None = None,
    thr_name: str = "youden",
) -> dict:
    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob.astype(float), 1e-7, 1 - 1e-7)
    if thr is None:
        thr = youden_threshold(y_true, y_prob)
    y_hat = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "n": int(len(y_true)),
        "event_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "ece_10": expected_calibration_error(y_true, y_prob, 10),
        "threshold": float(thr),
        "threshold_rule": thr_name,
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "sensitivity_recall": float(sens),
        "specificity": float(spec),
        "ppv_precision": float(ppv),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


class StayDS(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class MortalityMLP(nn.Module):
    def __init__(self, d: int, dropout: float = 0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TabularTransformer(nn.Module):
    def __init__(self, d: int, h: int = 64, layers: int = 1, dropout: float = 0.4):
        super().__init__()
        self.proj = nn.Linear(1, h)
        self.pos = nn.Parameter(torch.randn(1, d, h) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=4,
            dim_feedforward=h * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, x):
        h = self.proj(x.unsqueeze(-1)) + self.pos
        h = self.enc(h).mean(1)
        return self.head(h).squeeze(-1)


def prep_xy(df: pd.DataFrame, cols: list[str]):
    x = df[cols].copy()
    y = df["HOSPITAL_MORTALITY"].astype(int)
    tr = df["split"] == "train"
    va = df["split"] == "val"
    te = df["split"] == "test"
    xt, xv, xe = x.loc[tr].copy(), x.loc[va].copy(), x.loc[te].copy()
    yt, yv, ye = y.loc[tr], y.loc[va], y.loc[te]
    fill = xt.median(numeric_only=True).fillna(0.0)
    xt, xv, xe = xt.fillna(fill), xv.fillna(fill), xe.fillna(fill)
    mu, sd = xt.mean(0), xt.std(0, ddof=0).replace(0, 1).fillna(1)
    xt = (xt - mu) / sd
    xv = (xv - mu) / sd
    xe = (xe - mu) / sd
    return (
        xt.to_numpy(np.float32),
        yt.to_numpy(int),
        xv.to_numpy(np.float32),
        yv.to_numpy(int),
        xe.to_numpy(np.float32),
        ye.to_numpy(int),
    )


@torch.no_grad()
def predict_dl(net, X, device, bs=512):
    net.eval()
    loader = DataLoader(StayDS(X, np.zeros(len(X))), batch_size=bs, shuffle=False)
    outs = []
    for xb, _ in loader:
        outs.append(torch.sigmoid(net(xb.to(device))).cpu().numpy())
    return np.concatenate(outs)


def train_dl(model_name: str, Xt, yt, Xv, yv, device):
    set_seed(SEED)
    d = Xt.shape[1]
    if model_name == "mlp":
        net = MortalityMLP(d).to(device)
        lr, wd, epochs, patience = 5e-4, 5e-3, 100, 15
    else:
        net = TabularTransformer(d).to(device)
        lr, wd, epochs, patience = 3e-4, 1e-2, 100, 15

    n_pos = max(float(yt.sum()), 1.0)
    n_neg = float(len(yt) - yt.sum())
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / n_pos], device=device)
    )
    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    tr_loader = DataLoader(StayDS(Xt, yt.astype(np.float32)), batch_size=BATCH_SIZE, shuffle=True)
    best_auc, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        net.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        p_va = predict_dl(net, Xv, device)
        va_auc = roc_auc_score(yv, p_va)
        if va_auc > best_auc + 1e-4:
            best_auc, bad = va_auc, 0
            best_state = copy.deepcopy({k: v.cpu().clone() for k, v in net.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                print(f"  {model_name}: early stop ep={ep} best_val_auc={best_auc:.4f}")
                break
    net.load_state_dict(best_state)
    net.to(device)
    return net


def main():
    print("=" * 60)
    print("10: Rich metrics comparison (HistGBM / MLP / Transformer)")
    print("=" * 60)
    path = PROJ / "data" / "processed" / "stay_features_48h.parquet"
    df = pd.read_parquet(path)
    ycol = "HOSPITAL_MORTALITY"

    # --- HistGBM: all numeric cols like 08 ---
    drop = {"ICUSTAY_ID", "SUBJECT_ID", "split", "GENDER", "ADMISSION_TYPE", ycol}
    tree_cols = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"HistGBM features: {len(tree_cols)} | DL features: {len(FEATURE_COLS_V1)}")

    def split_raw(cols):
        tr = df["split"] == "train"
        va = df["split"] == "val"
        te = df["split"] == "test"
        X = df[cols].astype(float)
        y = df[ycol].astype(int)
        return X.loc[tr], y.loc[tr], X.loc[va], y.loc[va], X.loc[te], y.loc[te]

    Xtr, ytr, Xva, yva, Xte, yte = split_raw(tree_cols)
    print("  fitting HistGBM ...")
    clf = HistGradientBoostingClassifier(
        max_depth=2,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=5.0,
        learning_rate=0.03,
        max_bins=128,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=25,
        random_state=SEED,
    )
    clf.fit(Xtr, ytr)
    # threshold from VAL only
    p_va_tree = clf.predict_proba(Xva)[:, 1]
    thr_tree = youden_threshold(yva.to_numpy(), p_va_tree)

    rows = []
    for split, X, y in [
        ("train", Xtr, ytr),
        ("val", Xva, yva),
        ("test", Xte, yte),
    ]:
        p = clf.predict_proba(X)[:, 1]
        m = rich_metrics(y.to_numpy(), p, thr=thr_tree, thr_name="youden_on_val")
        m["model"] = "HistGBM"
        m["split"] = split
        m["n_features"] = len(tree_cols)
        rows.append(m)
        print(f"  HistGBM {split}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
              f"F1={m['f1']:.3f} Sens={m['sensitivity_recall']:.3f} Spec={m['specificity']:.3f}")

    # --- DL models on FEATURE_COLS_V1 ---
    Xt, yt, Xv, yv, Xe, ye = prep_xy(df, FEATURE_COLS_V1)
    for name in ["mlp", "transformer"]:
        print(f"  training {name} ...")
        net = train_dl(name, Xt, yt, Xv, yv, DEVICE)
        p_va = predict_dl(net, Xv, DEVICE)
        thr = youden_threshold(yv, p_va)
        for split, X, y in [
            ("train", Xt, yt),
            ("val", Xv, yv),
            ("test", Xe, ye),
        ]:
            p = predict_dl(net, X, DEVICE)
            m = rich_metrics(y, p, thr=thr, thr_name="youden_on_val")
            m["model"] = name.upper() if name == "mlp" else "Transformer"
            m["split"] = split
            m["n_features"] = len(FEATURE_COLS_V1)
            rows.append(m)
            print(
                f"  {m['model']} {split}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
                f"F1={m['f1']:.3f} Sens={m['sensitivity_recall']:.3f} Spec={m['specificity']:.3f}"
            )

    out = pd.DataFrame(rows)
    # column order
    cols = [
        "model", "split", "n", "n_features", "event_rate",
        "auroc", "auprc", "brier", "log_loss", "ece_10",
        "threshold", "threshold_rule",
        "accuracy", "f1", "sensitivity_recall", "specificity",
        "ppv_precision", "npv", "tp", "fp", "tn", "fn",
    ]
    out = out[cols]
    out_dir = PROJ / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    path_out = out_dir / "baseline_models_rich_metrics.csv"
    out.to_csv(path_out, index=False)

    # pretty test-only table
    test = out[out["split"] == "test"].copy()
    print("\n" + "=" * 70)
    print("TEST SET COMPARISON")
    print("=" * 70)
    show = [
        "model", "auroc", "auprc", "brier", "log_loss", "ece_10",
        "f1", "sensitivity_recall", "specificity", "ppv_precision", "npv",
    ]
    print(test[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {path_out}")
    print("10 done.")


if __name__ == "__main__":
    main()
