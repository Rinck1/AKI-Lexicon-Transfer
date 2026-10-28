#!/usr/bin/env python3
"""Train-only discretization of 11 lab stage states (docs/20 §3.2 / §9.1).

Writes lab thresholds and lab/motif atoms. Never reads test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.graph_constants import A_MIN_EVENT  # noqa: E402
from scripts.utils.graph_sparse import (  # noqa: E402
    build_sparse_matrix_for_stays,
    sparse_feature_names,
)
from scripts.utils.path_atoms import (  # noqa: E402
    fit_lab_thresholds,
    lab_atoms_from_raw,
    motif_atoms_from_events,
)
from scripts.utils.path_protocol import refuse_test_path, refuse_test_split  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

DEFAULT_EVENTS = PROJ / "data/processed/shapelet/events/events_final_7var_motif84"
DEFAULT_OUT = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", default=str(DEFAULT_EVENTS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--max-stays", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ev_dir = Path(args.events)
    out_dir = Path(args.out)
    refuse_test_path(ev_dir)
    refuse_test_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.read_parquet(ev_dir / "stay_index.parquet")
    for sp in idx["split"].astype(str).unique():
        refuse_test_split(sp)
    events = pd.read_parquet(ev_dir / "activations_long.parquet")
    for sp in events["split"].astype(str).unique():
        refuse_test_split(sp)

    train_ids = idx.loc[idx["split"] == "train", "ICUSTAY_ID"].astype(int).tolist()
    val_ids = idx.loc[idx["split"] == "val", "ICUSTAY_ID"].astype(int).tolist()
    if args.max_stays:
        train_ids = train_ids[: int(args.max_stays)]
        val_ids = val_ids[: int(args.max_stays)]
        keep = set(train_ids) | set(val_ids)
        idx = idx[idx["ICUSTAY_ID"].isin(keep)]
        events = events[events["ICUSTAY_ID"].isin(keep)]

    _log(f"[21] train={len(train_ids)} val={len(val_ids)}")
    motif = motif_atoms_from_events(events, a_min=A_MIN_EVENT)
    motif.to_parquet(out_dir / "motif_atoms.parquet", index=False)
    _log(f"[21] motif atoms={len(motif)} stays={motif['ICUSTAY_ID'].nunique()}")

    _log("[21] loading hourly grid (train-only thresholds)…")
    grid, stays = load_shapelet_tables(DATA_PROCESSED)
    roster = idx[["ICUSTAY_ID", "SUBJECT_ID", "split", "HOSPITAL_MORTALITY"]].copy()
    roster = roster.drop_duplicates("ICUSTAY_ID")
    all_ids = roster["ICUSTAY_ID"].astype(int).tolist()
    raw = build_sparse_matrix_for_stays(grid, all_ids, show_progress=True)
    names = sparse_feature_names()
    thr = fit_lab_thresholds(raw, np.array(all_ids), train_ids, names=names)
    (out_dir / "lab_thresholds.json").write_text(
        json.dumps(thr, indent=2), encoding="utf-8"
    )
    lab = lab_atoms_from_raw(raw, roster, thr, names=names)
    lab.to_parquet(out_dir / "lab_atoms.parquet", index=False)
    _log(f"[21] lab atoms={len(lab)} stays={lab['ICUSTAY_ID'].nunique() if len(lab) else 0}")

    atoms = pd.concat([motif, lab], ignore_index=True)
    atoms.to_parquet(out_dir / "atoms.parquet", index=False)
    manifest = {
        "stage": "path_lab_discretize",
        "test_read": False,
        "events": str(ev_dir.resolve()),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_motif_atoms": int(len(motif)),
        "n_lab_atoms": int(len(lab)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "a_min": A_MIN_EVENT,
    }
    (out_dir / "discretize_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _log("[21] DONE")


if __name__ == "__main__":
    main()
