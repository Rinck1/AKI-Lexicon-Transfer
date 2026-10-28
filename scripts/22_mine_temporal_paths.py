#!/usr/bin/env python3
"""Mine 2–4 step temporal paths on train only (docs/20 §4 / §9.2)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.path_atoms import atoms_by_stay  # noqa: E402
from scripts.utils.path_mine import extend_from_parents, frequent_paths, score_paths  # noqa: E402
from scripts.utils.path_protocol import (  # noqa: E402
    PATH_GROW_PARENT_TOP,
    PATH_SEED,
    min_path_support,
    parse_path_key,
    refuse_test_path,
    refuse_test_split,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402

DEFAULT_IN = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms-dir", default=str(DEFAULT_IN))
    p.add_argument("--out", default=str(DEFAULT_IN))
    p.add_argument(
        "--extend",
        action="store_true",
        help="Grow length 3–4 from top readable length-2 parents; merge, do not rescore 2-step.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.atoms_dir)
    out = Path(args.out)
    refuse_test_path(src)
    refuse_test_path(out)
    out.mkdir(parents=True, exist_ok=True)

    atoms = pd.read_parquet(src / "atoms.parquet")
    for sp in atoms["split"].astype(str).unique():
        refuse_test_split(sp)
    train_atoms = atoms[atoms["split"] == "train"].copy()
    train_ids = sorted(int(x) for x in train_atoms["ICUSTAY_ID"].unique())
    # include train stays with zero atoms from stay index if present
    idx_path = Path(str(src).replace("paths", "shapelet/events"))
    ev_idx = PROJ / "data/processed/shapelet/events/events_final_7var_motif84/stay_index.parquet"
    if ev_idx.is_file():
        idx = pd.read_parquet(ev_idx)
        refuse_test_path(ev_idx)
        train_ids = sorted(
            int(x) for x in idx.loc[idx["split"] == "train", "ICUSTAY_ID"].tolist()
        )

    n_train = len(train_ids)
    thr = min_path_support(n_train)
    _log(f"[22] train stays={n_train} min_support={thr} train_atoms={len(train_atoms)}")

    stay_atoms = atoms_by_stay(train_atoms)
    grow_n = 0
    parent_keys: list[str] = []
    if args.extend:
        prev = pd.read_parquet(src / "path_stable_train.parquet")
        work = prev[prev["readable"].astype(bool)].copy()
        work["_abs"] = work["logit_coef"].abs()
        work = work.sort_values("_abs", ascending=False).head(PATH_GROW_PARENT_TOP)
        parents = {parse_path_key(p) for p in work["path"]}
        parent_keys = work["path"].tolist()
        grow_n = len(parents)
        _log(f"[22] extending from {grow_n} readable length-2 parents…")
        freq3 = extend_from_parents(stay_atoms, train_ids, parents, min_support=thr)
        freq4 = extend_from_parents(stay_atoms, train_ids, set(freq3), min_support=thr)
        freq = {**freq3, **freq4}
        _log(f"[22] frequent grown paths={len(freq)} (len3={len(freq3)} len4={len(freq4)})")
    else:
        _log("[22] counting frequent paths (Apriori)…")
        freq = frequent_paths(stay_atoms, train_ids, min_support=thr)
        _log(f"[22] frequent paths={len(freq)}")

    _grid, stays = load_shapelet_tables(DATA_PROCESSED)
    stay_idx = stays.set_index("ICUSTAY_ID")
    y_by = {}
    subj_of = {}
    static_by = {}
    meta_src = ev_idx if ev_idx.is_file() else None
    if meta_src:
        meta = pd.read_parquet(meta_src)
        meta = meta[meta["split"] == "train"]
        for rec in meta.itertuples(index=False):
            sid = int(rec.ICUSTAY_ID)
            y_by[sid] = float(rec.HOSPITAL_MORTALITY)
            subj_of[sid] = int(rec.SUBJECT_ID)
    for sid in train_ids:
        row = stay_idx.loc[int(sid)]
        if sid not in y_by:
            y_by[sid] = float(row["HOSPITAL_MORTALITY"])
            subj_of[sid] = int(row["SUBJECT_ID"])
        static_by[sid] = {c: float(row[c]) for c in SHAPELET_STATIC_COLS}

    _log("[22] scoring + bootstrap (train only)…")
    scored = score_paths(
        freq,
        train_ids,
        y_by,
        subj_of,
        static_by,
        list(SHAPELET_STATIC_COLS),
        min_support=thr,
        seed=PATH_SEED,
    )
    cand = pd.DataFrame(scored)
    if args.extend:
        old = pd.read_parquet(out / "path_candidates_train.parquet")
        if len(cand) and "path" in cand.columns:
            old = old[~old["path"].isin(set(cand["path"].tolist()))]
            cand = pd.concat([old, cand], ignore_index=True)
        else:
            cand = old
    cand.to_parquet(out / "path_candidates_train.parquet", index=False)
    cand.to_csv(out / "path_candidates_train.csv", index=False, encoding="utf-8-sig")
    stable = cand[cand["stable"]].copy()
    stable.to_parquet(out / "path_stable_train.parquet", index=False)
    stable.to_csv(out / "path_stable_train.csv", index=False, encoding="utf-8-sig")
    n_stable = int(len(stable))
    n_read = int(stable["readable"].sum()) if n_stable else 0
    by_len = (
        stable.groupby("n_atoms").size().astype(int).to_dict() if n_stable else {}
    )
    _log(f"[22] stable={n_stable} readable={n_read} by_len={by_len}")
    summary = {
        "stage": "path_mine_train_extend" if args.extend else "path_mine_train",
        "test_read": False,
        "val_used": False,
        "n_train": n_train,
        "min_support": thr,
        "n_frequent": int(len(cand)),
        "n_stable": n_stable,
        "n_stable_readable": n_read,
        "n_stable_by_len": {str(k): int(v) for k, v in by_len.items()},
        "grow_parent_top": grow_n if args.extend else 0,
        "grow_parents": parent_keys if args.extend else [],
        "seed": PATH_SEED,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "mine_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log("[22] DONE")


if __name__ == "__main__":
    main()
