#!/usr/bin/env python3
"""eICU stay-level static table for frozen Shapelet / path logistic.

split is always 'external'. Missing comorbidities stay 0. No eICU median fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.eicu_io import load_eicu, out_dir, save_eicu  # noqa: E402
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    primary = load_eicu("aki_primary_ha48")
    base = load_eicu("icu_base")
    comorbid = load_eicu("meas_comorbid")
    feats = primary[
        [
            "ICUSTAY_ID",
            "SUBJECT_ID",
            "HOSPITAL_MORTALITY",
            "AGE",
            "GENDER",
            "onset_stage",
            "onset_hour_from_icu",
        ]
    ].copy()
    feats["split"] = "external"
    feats["GENDER_M"] = (feats["GENDER"].astype(str) == "M").astype(int)
    ccols = [c for c in comorbid.columns if c not in ("SUBJECT_ID", "HADM_ID")]
    feats = feats.merge(comorbid[ccols], on="ICUSTAY_ID", how="left")
    for c in SHAPELET_STATIC_COLS:
        if c not in feats.columns:
            feats[c] = 0.0
        feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0)
    path = save_eicu(feats, "stay_features_48h")
    # alias expected by 28_external_paths_stub
    save_eicu(feats, "stay_features")
    miss = [c for c in SHAPELET_STATIC_COLS if c not in feats.columns]
    if miss:
        raise KeyError(miss)
    (out_dir() / "stage44_manifest.json").write_text(
        json.dumps(
            {
                "stage": "44_eicu_stay_features",
                "n": int(len(feats)),
                "static_cols": list(SHAPELET_STATIC_COLS),
                "mortality": float(feats["HOSPITAL_MORTALITY"].mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(f"[44] stays={len(feats)} → {path}")


if __name__ == "__main__":
    main()
