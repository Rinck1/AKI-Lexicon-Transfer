#!/usr/bin/env python3
"""
Build patient evolution graphs from Shapelet events (docs/13 §5, §7, §10.2).

- Train-only edge vocabulary (support filter)
- val maps to frozen vocab only (drop unseen)
- Graph packs for train/val; never writes graph_pack_test.pt by default
- Includes a_dense, times, sparse (308), static (11)

Usage:
  python -u scripts/16_build_evolution_graphs.py ^
    --events data/processed/shapelet/events/events_final_7var_L4_6_12_K3 ^
    --run-name graph_final_7var_L4_6_12_K3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import DATA_PROCESSED  # noqa: E402
from scripts.utils.graph_builder import (  # noqa: E402
    build_canonical_edges_for_patient,
    build_graphs_for_split,
    collect_train_edge_support,
    empty_graph_rate,
    freeze_edge_vocabulary,
)
from scripts.utils.graph_constants import (  # noqa: E402
    A_MIN_EVENT,
    CO_OCCUR_MAX_HOURS,
    EDGE_TYPE_CO_OCCUR,
    EDGE_TYPE_PRECEDES,
    MAX_CANONICAL_EDGES,
    MAX_CO_OCCUR_NEIGHBORS,
    MAX_PRECEDES_OUT,
    N_EDGE_ATTR,
    N_SPARSE_FEATURES,
    N_STATIC_FEATURES,
    PRECEDES_MAX_HOURS,
    PRECEDES_MIN_HOURS,
    T_HOURS_GRAPH,
    min_edge_support,
)
from scripts.utils.graph_data import (  # noqa: E402
    patient_graph_to_tensors,
    pad_edges,
    pack_dict_from_patients,
)
from scripts.utils.graph_events import select_active_events  # noqa: E402
from scripts.utils.graph_sparse import (  # noqa: E402
    build_sparse_matrix_for_stays,
    fit_sparse_scaler,
    save_sparse_scaler,
    sparse_feature_names,
)
from scripts.utils.shapelet_constants import SHAPELET_STATIC_COLS  # noqa: E402
from scripts.utils.shapelet_prep import load_shapelet_tables  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_static_from_frozen(events_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    frozen = events_dir / "shapelet_frozen_qp50.pt"
    if not frozen.is_file():
        raise FileNotFoundError(f"need {frozen} from script 15")
    obj = torch.load(frozen, map_location="cpu", weights_only=False)
    sc = obj["static_scaler"]
    cols = list(sc.get("cols") or SHAPELET_STATIC_COLS)
    mean = np.asarray(sc["mean"], dtype=np.float64)
    std = np.asarray(sc["std"], dtype=np.float64)
    return cols, mean, std


def transform_static_row(
    row: pd.Series,
    cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    raw = []
    for c in cols:
        v = row[c] if c in row.index else np.nan
        raw.append(float(v) if pd.notna(v) else float("nan"))
    x = np.asarray(raw, dtype=np.float64)
    nan = np.isnan(x)
    x[nan] = mean[nan]
    sd = np.maximum(std, 1e-6)
    return ((x - mean) / sd).astype(np.float32)


def dense_from_long(
    stay_df: pd.DataFrame, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.zeros(n_nodes, dtype=np.float32)
    nt = np.zeros(n_nodes, dtype=np.float32)
    ntw = np.zeros(n_nodes, dtype=np.float32)
    nL = np.zeros(n_nodes, dtype=np.float32)
    for r in stay_df.itertuples():
        i = int(r.node_id)
        if 0 <= i < n_nodes:
            a[i] = float(r.a)
            nt[i] = float(getattr(r, "t_star_norm", float(r.t_star) / T_HOURS_GRAPH))
            ntw[i] = float(getattr(r, "t_within_stage", 0.0))
            L_val = getattr(r, "L", None)
            if L_val is not None and pd.notna(L_val):
                nL[i] = float(L_val) / 12.0
    return a, nt, ntw, nL


def edges_to_records(graphs) -> pd.DataFrame:
    rows = []
    for g in graphs:
        for e in g.edges:
            rows.append(
                {
                    "src": e.src,
                    "dst": e.dst,
                    "edge_type": e.edge_type,
                    "train_support": e.train_support,
                    "edge_type_name": (
                        "CO_OCCUR" if e.edge_type == EDGE_TYPE_CO_OCCUR else "PRECEDES"
                    ),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build evolution graph packs")
    p.add_argument(
        "--events",
        type=str,
        required=True,
        help="directory with activations_long.parquet + node_dictionary + frozen qp50",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="output under data/processed/graphs/<run-name>",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-stays",
        type=int,
        default=None,
        help="debug: limit stays per split after loading events",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events_dir = Path(args.events)
    if not events_dir.is_dir():
        raise FileNotFoundError(events_dir)

    long_path = events_dir / "activations_long.parquet"
    nd_path = events_dir / "node_dictionary.parquet"
    if not long_path.is_file():
        raise FileNotFoundError(long_path)
    if not nd_path.is_file():
        raise FileNotFoundError(nd_path)

    run_name = args.run_name or f"graph_{events_dir.name}"
    out_dir = PROJ / "data" / "processed" / "graphs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("=" * 60)
    _log("16: build evolution graphs")
    _log("=" * 60)
    _log(f"  events={events_dir}")
    _log(f"  out={out_dir}")

    nd = pd.read_parquet(nd_path)
    n_nodes = int(nd["node_id"].max()) + 1
    long_df = pd.read_parquet(long_path)
    if "test" in set(long_df["split"].unique()):
        _log("[warn] events contain test rows — they will be ignored (no test pack)")
        long_df = long_df.loc[long_df["split"] != "test"].copy()

    # copy node dictionary + event policy
    nd.to_parquet(out_dir / "node_dictionary.parquet", index=False)
    (out_dir / "node_dictionary.json").write_text(
        json.dumps(nd.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    for name in (
        "event_policy.json",
        "export_manifest.json",
        "template_family_mapping.parquet",
    ):
        src = events_dir / name
        if src.is_file():
            if src.suffix == ".parquet":
                pd.read_parquet(src).to_parquet(out_dir / name, index=False)
            else:
                (out_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    event_policy = {
        "a_min": A_MIN_EVENT,
        "co_occur_max_h": CO_OCCUR_MAX_HOURS,
        "precedes_min_h": PRECEDES_MIN_HOURS,
        "precedes_max_h": PRECEDES_MAX_HOURS,
        "max_co_neighbors": MAX_CO_OCCUR_NEIGHBORS,
        "max_precedes_out": MAX_PRECEDES_OUT,
        "max_canonical_edges": MAX_CANONICAL_EDGES,
        "n_edge_attr": N_EDGE_ATTR,
    }
    (out_dir / "edge_policy.json").write_text(
        json.dumps(event_policy, indent=2), encoding="utf-8"
    )

    # active events
    active = select_active_events(long_df, a_min=A_MIN_EVENT)
    _log(
        f"[16] active events: {len(active)}  "
        f"stays={active['ICUSTAY_ID'].nunique() if len(active) else 0}"
    )

    stay_index_path = events_dir / "stay_index.parquet"
    if stay_index_path.is_file():
        stay_roster = pd.read_parquet(stay_index_path)
        if "test" in set(stay_roster["split"].astype(str)):
            stay_roster = stay_roster.loc[stay_roster["split"].astype(str) != "test"].copy()
        train_ids = (
            stay_roster.loc[stay_roster["split"] == "train", "ICUSTAY_ID"]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )
        val_ids = (
            stay_roster.loc[stay_roster["split"] == "val", "ICUSTAY_ID"]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )
        _log(
            f"[16] stay_index: train={len(train_ids)} val={len(val_ids)} "
            f"(includes empty-event stays)"
        )
    else:
        train_ids = (
            long_df.loc[long_df["split"] == "train", "ICUSTAY_ID"]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )
        val_ids = (
            long_df.loc[long_df["split"] == "val", "ICUSTAY_ID"]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )
    if args.max_stays is not None:
        train_ids = train_ids[: args.max_stays]
        val_ids = val_ids[: args.max_stays]
        long_df = long_df[long_df["ICUSTAY_ID"].isin(set(train_ids) | set(val_ids))]
        active = active[active["ICUSTAY_ID"].isin(set(train_ids) | set(val_ids))]

    n_train = len(train_ids)
    thr = min_edge_support(n_train)
    _log(f"[16] n_train_stays={n_train}  min_edge_support={thr}")

    # train edge lists
    train_edge_lists = []
    for sid in tqdm(train_ids, desc="train edges", unit="stay"):
        ev = active.loc[active["ICUSTAY_ID"] == sid]
        train_edge_lists.append(build_canonical_edges_for_patient(ev))
    support = collect_train_edge_support(train_edge_lists)
    vocab = freeze_edge_vocabulary(support, n_train_stays=n_train)
    _log(
        f"[16] edge support types={len(support)}  vocab_kept={len(vocab)}  "
        f"drop_ratio={1 - len(vocab) / max(len(support), 1):.3f}"
    )

    vocab_rows = [
        {
            "src": k[0],
            "dst": k[1],
            "edge_type": k[2],
            "edge_type_name": "CO_OCCUR" if k[2] == EDGE_TYPE_CO_OCCUR else "PRECEDES",
            "train_support": s,
        }
        for k, s in sorted(vocab.items())
    ]
    vocab_df = pd.DataFrame(vocab_rows)
    vocab_df.to_parquet(out_dir / "edge_vocabulary_train.parquet", index=False)

    # stay meta + static
    _grid, stays = load_shapelet_tables(DATA_PROCESSED)
    stay_idx = stays.set_index("ICUSTAY_ID")
    static_cols, st_mean, st_std = load_static_from_frozen(events_dir)

    # sparse features
    _log("[16] building sparse stage features (train fit)…")
    train_raw = build_sparse_matrix_for_stays(_grid, train_ids, show_progress=True)
    sparse_scaler = fit_sparse_scaler(train_raw)
    save_sparse_scaler(out_dir / "sparse_scaler_train.json", sparse_scaler)
    (out_dir / "sparse_feature_names.json").write_text(
        json.dumps(sparse_feature_names(), indent=2), encoding="utf-8"
    )
    train_sparse = np.stack([sparse_scaler.transform(r) for r in train_raw], axis=0)

    _log("[16] building sparse for val…")
    val_raw = build_sparse_matrix_for_stays(_grid, val_ids, show_progress=True)
    val_sparse = np.stack([sparse_scaler.transform(r) for r in val_raw], axis=0)

    train_sparse_map = {int(sid): train_sparse[i] for i, sid in enumerate(train_ids)}
    val_sparse_map = {int(sid): val_sparse[i] for i, sid in enumerate(val_ids)}

    def pack_split(split: str, ids: list[int], apply_vocab: bool) -> tuple[list, dict]:
        meta = []
        for sid in ids:
            row = stay_idx.loc[sid]
            meta.append(
                {
                    "ICUSTAY_ID": sid,
                    "SUBJECT_ID": int(row["SUBJECT_ID"]),
                    "split": split,
                    "HOSPITAL_MORTALITY": float(row["HOSPITAL_MORTALITY"]),
                }
            )
        meta_df = pd.DataFrame(meta)
        act_split = active.loc[active["ICUSTAY_ID"].isin(ids)]
        graphs = build_graphs_for_split(
            long_df.loc[long_df["ICUSTAY_ID"].isin(ids)],
            act_split,
            edge_vocab=vocab,
            n_train_stays=n_train,
            apply_vocab=apply_vocab,
            stay_meta=meta_df,
        )
        patients = []
        dropped_total = 0
        for g in tqdm(graphs, desc=f"pack {split}", unit="stay"):
            sid = g.stay_id
            stay_long = long_df.loc[long_df["ICUSTAY_ID"] == sid]
            a_dense, nt, ntw, nL = dense_from_long(stay_long, n_nodes)
            if split == "train":
                sp = train_sparse_map[sid]
            else:
                sp = val_sparse_map[sid]
            st_row = stay_idx.loc[sid]
            static = transform_static_row(st_row, static_cols, st_mean, st_std)
            sample = patient_graph_to_tensors(
                g,
                a_dense=a_dense,
                node_time=nt,
                node_time_within_stage=ntw,
                sparse=sp,
                static=static,
                n_train_stays=n_train,
            )
            padded = pad_edges(sample)
            # L/12 on the winning event; 0 if the node is inactive. Unused by
            # the current GraphRiskModel; stored so 84-node training can add a
            # length channel without rebuilding packs.
            padded["node_length"] = torch.as_tensor(nL, dtype=torch.float32)
            patients.append(padded)
            dropped_total += g.dropped_unseen_edge

        stats = {
            "n_stays": len(graphs),
            "empty_graph_rate": empty_graph_rate(graphs),
            "mean_events": float(np.mean([g.n_events for g in graphs])) if graphs else 0.0,
            "mean_edges": float(np.mean([g.n_edges for g in graphs])) if graphs else 0.0,
            "dropped_unseen_edge_total": int(dropped_total),
            "dropped_unseen_per_stay": float(dropped_total / max(len(graphs), 1)),
            "mortality_rate": float(meta_df["HOSPITAL_MORTALITY"].mean()),
        }
        return patients, stats

    train_patients, train_stats = pack_split("train", train_ids, apply_vocab=True)
    val_patients, val_stats = pack_split("val", val_ids, apply_vocab=True)

    train_pack = pack_dict_from_patients(
        train_patients,
        meta={
            "split": "train",
            "n_nodes": n_nodes,
            "stats": train_stats,
            "test_built": False,
        },
    )
    val_pack = pack_dict_from_patients(
        val_patients,
        meta={
            "split": "val",
            "n_nodes": n_nodes,
            "stats": val_stats,
            "test_built": False,
        },
    )
    torch.save(train_pack, out_dir / "graph_pack_train.pt")
    torch.save(val_pack, out_dir / "graph_pack_val.pt")
    _log(f"[16] wrote graph_pack_train.pt ({len(train_patients)} stays)")
    _log(f"[16] wrote graph_pack_val.pt ({len(val_patients)} stays)")
    _log(f"  train stats: {train_stats}")
    _log(f"  val stats: {val_stats}")

    # hashes
    vocab_hash = sha256_file(out_dir / "edge_vocabulary_train.parquet")
    nd_hash = sha256_file(out_dir / "node_dictionary.parquet")
    frozen = events_dir / "shapelet_frozen_qp50.pt"
    frozen_hash = sha256_file(frozen) if frozen.is_file() else None
    train_subj = sorted(
        set(
            int(stay_idx.loc[sid, "SUBJECT_ID"])
            for sid in train_ids
            if sid in stay_idx.index
        )
    )
    train_subj_hash = sha256_bytes(
        json.dumps(train_subj, separators=(",", ":")).encode("utf-8")
    )

    # subject leakage
    val_subj = set(int(stay_idx.loc[sid, "SUBJECT_ID"]) for sid in val_ids)
    train_subj_set = set(train_subj)
    overlap = train_subj_set & val_subj
    if overlap:
        raise RuntimeError(f"SUBJECT_ID train/val overlap: {len(overlap)}")

    node_scheme = (
        "var_motif_family_stage"
        if "motif_family" in nd.columns
        else "var_L_k_stage"
    )
    mapping_src = events_dir / "template_family_mapping.parquet"
    manifest = {
        "run_name": run_name,
        "events_dir": str(events_dir.resolve()),
        "node_scheme": node_scheme,
        "n_nodes": n_nodes,
        "n_sparse": N_SPARSE_FEATURES,
        "n_static": N_STATIC_FEATURES,
        "edge_policy": event_policy,
        "min_edge_support": thr,
        "n_train_stays": n_train,
        "n_val_stays": len(val_ids),
        "vocab_size": len(vocab),
        "edge_vocabulary_hash": vocab_hash,
        "node_dictionary_hash": nd_hash,
        "shapelet_frozen_hash": frozen_hash,
        "train_subject_hash": train_subj_hash,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "template_family_mapping_hash": (
            sha256_file(mapping_src) if mapping_src.is_file() else None
        ),
        "test_built": False,
        "paper_eligible": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
    }
    (out_dir / "graph_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _log("[16] DONE")
    _log(f"  manifest: {out_dir / 'graph_manifest.json'}")
    _log("  next stage D (needs approval): 17_train_gnn_risk.py --train")


if __name__ == "__main__":
    main()
