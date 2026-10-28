#!/usr/bin/env python3
"""Apply frozen MIMIC Shapelet q + family map + lab thresholds to eICU.

Never refits q, family geometry, or Δ medians. Scalers come from MIMIC train.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.eicu_constants import MIMIC_PATHS_DIR  # noqa: E402
from scripts.utils.eicu_io import load_eicu, out_dir, save_eicu  # noqa: E402
from scripts.utils.graph_constants import A_MIN_EVENT  # noqa: E402
from scripts.utils.graph_events import select_active_events  # noqa: E402
from scripts.utils.graph_sparse import build_sparse_matrix_for_stays, sparse_feature_names  # noqa: E402
from scripts.utils.path_atoms import lab_atoms_from_raw, motif_atoms_from_events  # noqa: E402
from scripts.utils.path_protocol import sha256_canonical  # noqa: E402
from scripts.utils.shapelet_constants import BATCH_STAYS, SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_family import remap_activation_long  # noqa: E402
from scripts.utils.shapelet_prep import (  # noqa: E402
    build_bundle,
    fit_static_scaler,
    fit_value_standardizer,
    load_shapelet_tables,
)

_spec15 = importlib.util.spec_from_file_location(
    "export15", PROJ / "scripts" / "15_export_shapelet_events.py"
)
_mod15 = importlib.util.module_from_spec(_spec15)
assert _spec15.loader is not None
_spec15.loader.exec_module(_mod15)
export_split_activations = _mod15.export_split_activations
load_frozen_model = _mod15.load_frozen_model

CKPT = PROJ / "data/processed/shapelet/final/final_7var_L4_6_12_K3/ckpt_best.pt"
FROZEN_Q = PROJ / "data/processed/shapelet/events/events_final_7var_L4_6_12_K3/shapelet_frozen_qp50.pt"
MOTIF_DIR = PROJ / "data/processed/shapelet/events/events_final_7var_motif84"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _export_motif(grid: pd.DataFrame, stays: pd.DataFrame, ext_ids: list[int]) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"[45] shapelet device={device} n={len(ext_ids)}")
    model, info = load_frozen_model(CKPT, device)
    blob = torch.load(FROZEN_Q, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state"])
    model.eval()
    model.freeze_templates(True)
    mimic_grid, mimic_stays = load_shapelet_tables(DATA_PROCESSED)
    train_ids = mimic_stays.loc[mimic_stays["split"] == "train", "ICUSTAY_ID"].astype(int).tolist()
    value_scaler = fit_value_standardizer(mimic_grid, train_ids, vars_=info["vars"])
    static_scaler = fit_static_scaler(mimic_stays, train_ids, cols=SHAPELET_STATIC_COLS)
    del mimic_grid
    bundle = build_bundle(
        grid,
        stays,
        value_scaler,
        static_scaler,
        vars_=info["vars"],
        static_cols=SHAPELET_STATIC_COLS,
        stay_ids=ext_ids,
        desc="prep eicu",
        show_progress=True,
    )
    raw = export_split_activations(
        model, bundle, device, BATCH_STAYS, info["activation_names"], split_label="external"
    )
    mapping = pd.read_parquet(MOTIF_DIR / "template_family_mapping.parquet")
    nodes = pd.read_parquet(MOTIF_DIR / "node_dictionary.parquet")
    remapped = remap_activation_long(raw, mapping, nodes, drop_test=False)
    winners = select_active_events(remapped, a_min=A_MIN_EVENT)
    winners = winners.copy()
    winners["is_active"] = True
    return motif_atoms_from_events(winners, a_min=A_MIN_EVENT)


def main() -> None:
    freeze = json.loads((MIMIC_PATHS_DIR / "path_freeze_manifest.json").read_text(encoding="utf-8"))
    thr = json.loads((MIMIC_PATHS_DIR / "lab_thresholds.json").read_text(encoding="utf-8"))
    digest = freeze["freeze_sha256"]
    _log(f"[45] freeze_sha256={digest}")

    grid = load_eicu("hourly_grid_48h")
    stays = load_eicu("stay_features_48h")
    stays["split"] = "external"
    ext_ids = sorted(int(x) for x in stays["ICUSTAY_ID"])
    motif = _export_motif(grid, stays, ext_ids)
    save_eicu(motif, "motif_atoms")

    roster = stays[["ICUSTAY_ID", "SUBJECT_ID", "split", "HOSPITAL_MORTALITY"]].drop_duplicates("ICUSTAY_ID")
    raw = build_sparse_matrix_for_stays(grid, ext_ids, show_progress=True)
    names = sparse_feature_names()
    lab = lab_atoms_from_raw(raw, roster, thr, names=names)
    save_eicu(lab, "lab_atoms")
    atoms = pd.concat([motif, lab], ignore_index=True)
    save_eicu(atoms, "atoms")
    (out_dir() / "freeze_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    (out_dir() / "stage45_manifest.json").write_text(
        json.dumps(
            {
                "stage": "45_eicu_frozen_atoms",
                "freeze_sha256": digest,
                "n_motif": int(len(motif)),
                "n_lab": int(len(lab)),
                "n_atoms": int(len(atoms)),
                "n_stays": int(len(ext_ids)),
                "refit_thresholds": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(f"[45] motif={len(motif)} lab={len(lab)} atoms={len(atoms)}")


if __name__ == "__main__":
    main()
