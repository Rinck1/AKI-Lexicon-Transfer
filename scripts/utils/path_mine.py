"""Apriori temporal paths, train bootstrap recurrence, and static-adjusted logit."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

from scripts.utils.path_protocol import (
    BOOT_RETAIN,
    BOOT_SUBJECT_FRAC,
    N_BOOT,
    PATH_MAX_LEN,
    PATH_MIN_LEN,
    PATH_SEED,
    is_readable,
    path_key,
)


def enumerate_stay_paths(
    atoms: list[tuple[int, str, bool]],
    *,
    min_len: int = PATH_MIN_LEN,
    max_len: int = PATH_MAX_LEN,
) -> list[tuple[str, ...]]:
    """Strictly increasing stage sequences; one atom per (stage, atom) pair."""
    by_stage: dict[int, list[str]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    for st, atom, _low in atoms:
        key = (int(st), str(atom))
        if key in seen:
            continue
        seen.add(key)
        by_stage[int(st)].append(str(atom))
    stages = sorted(by_stage)
    paths: list[tuple[str, ...]] = []

    def rec(start: int, chosen: list[str]) -> None:
        if min_len <= len(chosen) <= max_len:
            paths.append(tuple(chosen))
        if len(chosen) >= max_len:
            return
        for j in range(start, len(stages)):
            for atom in by_stage[stages[j]]:
                rec(j + 1, chosen + [atom])

    rec(0, [])
    return paths


def _stay_stage_atoms(
    atoms: list[tuple[int, str, bool]],
) -> dict[int, list[str]]:
    by_stage: dict[int, list[str]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    for st, atom, _low in atoms:
        key = (int(st), str(atom))
        if key in seen:
            continue
        seen.add(key)
        by_stage[int(st)].append(str(atom))
    return dict(by_stage)


def extend_from_parents(
    stay_atoms: dict[int, list[tuple[int, str, bool]]],
    stay_ids: list[int],
    parents: set[tuple[str, ...]],
    *,
    min_support: int,
) -> dict[tuple[str, ...], set[int]]:
    """One-atom extensions of frozen prefixes; child must still meet min_support."""
    allowed = {tuple(p) for p in parents}
    if not allowed:
        return {}
    buckets: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for sid in stay_ids:
        stages = _stay_stage_atoms(stay_atoms.get(int(sid), []))
        if not stages:
            continue
        loc: dict[str, int] = {}
        for st, atoms in stages.items():
            for atom in atoms:
                loc[atom] = int(st)
        have = set(loc)
        for pref in allowed:
            if any(a not in have for a in pref):
                continue
            ranks = [loc[a] for a in pref]
            if any(ranks[i] >= ranks[i + 1] for i in range(len(ranks) - 1)):
                continue
            last = ranks[-1]
            for st, atoms in stages.items():
                if int(st) <= last:
                    continue
                for atom in atoms:
                    buckets[pref + (atom,)].add(int(sid))
    return {p: sids for p, sids in buckets.items() if len(sids) >= int(min_support)}


def frequent_paths(
    stay_atoms: dict[int, list[tuple[int, str, bool]]],
    stay_ids: list[int],
    *,
    min_support: int,
    max_len: int = PATH_MIN_LEN,
    grow_parents: set[tuple[str, ...]] | None = None,
) -> dict[tuple[str, ...], set[int]]:
    """Apriori. Default is length-2; 3–4 only grow from explicit readable parents."""
    universe = [int(s) for s in stay_ids]
    per_stay = {
        sid: enumerate_stay_paths(stay_atoms.get(sid, []), min_len=2, max_len=2)
        for sid in universe
    }
    buckets: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for sid in universe:
        for p in per_stay[sid]:
            buckets[p].add(sid)
    frequent = {p: sids for p, sids in buckets.items() if len(sids) >= min_support}
    if int(max_len) <= 2:
        return frequent
    parents = set(grow_parents or ())
    if not parents:
        return frequent
    grown = extend_from_parents(stay_atoms, universe, parents, min_support=min_support)
    frequent.update(grown)
    if int(max_len) >= 4 and grown:
        grown4 = extend_from_parents(
            stay_atoms, universe, set(grown), min_support=min_support
        )
        frequent.update(grown4)
    return frequent


def _static_matrix(stays: dict[int, dict], stay_ids: list[int], cols: list[str]) -> np.ndarray:
    X = np.zeros((len(stay_ids), len(cols)), dtype=np.float64)
    for i, sid in enumerate(stay_ids):
        rec = stays[int(sid)]
        for j, c in enumerate(cols):
            X[i, j] = float(rec.get(c, 0.0))
    return X


def logistic_path_coef(
    y: np.ndarray,
    path_ind: np.ndarray,
    static: np.ndarray,
    *,
    l2: float = 1e-3,
) -> tuple[float, float, float]:
    """Return (coef, se, p_hat_unused) for the path column after intercept+static."""
    n = int(y.shape[0])
    X = np.column_stack([np.ones(n), path_ind.astype(np.float64), static])
    # drop near-constant static cols
    std = X.std(axis=0)
    keep = std > 1e-8
    keep[0] = True
    keep[1] = True
    X = X[:, keep]
    p = X.shape[1]
    w = np.zeros(p, dtype=np.float64)
    H = np.eye(p)
    for _ in range(40):
        z = np.clip(X @ w, -30.0, 30.0)
        ph = 1.0 / (1.0 + np.exp(-z))
        W = ph * (1.0 - ph) + 1e-8
        H = X.T @ (W[:, None] * X) + l2 * np.eye(p)
        g = X.T @ (ph - y) + l2 * w
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), float("nan")
        w = w - step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    try:
        se = float(np.sqrt(np.diag(np.linalg.inv(H))[1]))
    except np.linalg.LinAlgError:
        se = float("nan")
    return float(w[1]), se, float("nan")


def crude_assoc(y: np.ndarray, ind: np.ndarray) -> dict:
    y = y.astype(np.float64)
    ind = ind.astype(bool)
    n1 = int(ind.sum())
    n0 = int((~ind).sum())
    d1 = float(y[ind].mean()) if n1 else float("nan")
    d0 = float(y[~ind].mean()) if n0 else float("nan")
    # Woolf OR
    a = float(((ind) & (y > 0.5)).sum()) + 0.5
    b = float(((ind) & (y <= 0.5)).sum()) + 0.5
    c = float(((~ind) & (y > 0.5)).sum()) + 0.5
    d = float(((~ind) & (y <= 0.5)).sum()) + 0.5
    odds = (a / b) / (c / d)
    return {
        "n_path": n1,
        "n_other": n0,
        "death_rate_path": d1,
        "death_rate_other": d0,
        "death_rate_diff": (d1 - d0) if n1 and n0 else float("nan"),
        "odds_ratio": float(odds),
    }


def score_paths(
    frequent: dict[tuple[str, ...], set[int]],
    stay_ids: list[int],
    y_by_stay: dict[int, float],
    subject_of: dict[int, int],
    static_by_stay: dict[int, dict],
    static_cols: list[str],
    *,
    min_support: int,
    seed: int = PATH_SEED,
) -> list[dict]:
    stays = list(stay_ids)
    y = np.array([float(y_by_stay[s]) for s in stays], dtype=np.float64)
    static = _static_matrix(static_by_stay, stays, static_cols)
    mu = static.mean(axis=0)
    sd = np.maximum(static.std(axis=0), 1e-6)
    static_z = (static - mu) / sd
    stay_pos = {int(s): i for i, s in enumerate(stays)}

    subjects = np.array(sorted({int(subject_of[s]) for s in stays}), dtype=np.int64)
    stays_of: dict[int, list[int]] = defaultdict(list)
    for s in stays:
        stays_of[int(subject_of[s])].append(int(s))

    rng = np.random.default_rng(int(seed))
    boots: list[set[int]] = []
    k = max(1, int(round(BOOT_SUBJECT_FRAC * len(subjects))))
    for b in range(N_BOOT):
        pick = rng.choice(subjects, size=k, replace=False)
        chosen: set[int] = set()
        for subj in pick:
            chosen.update(stays_of[int(subj)])
        boots.append(chosen)

    from tqdm.auto import tqdm

    # Pass 1: support recurrence only (set intersection).
    rec_rate: dict[tuple[str, ...], float] = {}
    for path, sids in frequent.items():
        n_ok = sum(1 for chosen in boots if len(sids.intersection(chosen)) >= min_support)
        rec_rate[path] = n_ok / float(N_BOOT)
    survivors = {p: s for p, s in frequent.items() if rec_rate[p] >= BOOT_RETAIN}
    rows: list[dict] = []
    for path, sids in tqdm(list(survivors.items()), desc="score paths", unit="path"):
        ind = np.zeros(len(stays), dtype=np.float64)
        for sid in sids:
            ind[stay_pos[int(sid)]] = 1.0
        crude = crude_assoc(y, ind)
        coef, se, _ = logistic_path_coef(y, ind, static_z)
        if not np.isfinite(coef) or not np.isfinite(se) or se <= 0:
            ci_lo = ci_hi = float("nan")
            ci_ok = False
        else:
            ci_lo = coef - 1.96 * se
            ci_hi = coef + 1.96 * se
            ci_ok = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)
        sign = 0 if not np.isfinite(coef) or coef == 0 else (1 if coef > 0 else -1)

        # Bootstrap direction uses crude mortality difference (same sign as full-train logit).
        n_sign = 0
        if sign != 0:
            for chosen in boots:
                if len(sids.intersection(chosen)) < min_support:
                    continue
                ys = np.array([float(y_by_stay[s]) for s in chosen], dtype=np.float64)
                ins = np.array([s in sids for s in chosen], dtype=bool)
                if ins.sum() == 0 or (~ins).sum() == 0:
                    continue
                diff = float(ys[ins].mean() - ys[~ins].mean())
                if np.sign(diff) == sign:
                    n_sign += 1
        sign_rate = n_sign / float(N_BOOT)
        rr = rec_rate[path]
        stable = (
            len(sids) >= min_support
            and rr >= BOOT_RETAIN
            and sign_rate >= BOOT_RETAIN
            and ci_ok
            and sign != 0
        )
        rows.append(
            {
                "path": path_key(path),
                "n_atoms": len(path),
                "train_support": len(sids),
                "death_support": int(sum(y_by_stay[s] > 0.5 for s in sids)),
                **crude,
                "logit_coef": coef,
                "logit_se": se,
                "logit_ci_low": ci_lo,
                "logit_ci_high": ci_hi,
                "logit_sign": sign,
                "boot_support_rate": rr,
                "boot_sign_rate": sign_rate,
                "stable": bool(stable),
                "readable": is_readable(path),
            }
        )
    rows.sort(
        key=lambda r: (
            not r["stable"],
            r["n_atoms"],
            -r["train_support"],
            -abs(float(r["logit_coef"]) if np.isfinite(r["logit_coef"]) else 0.0),
        )
    )
    return rows


def path_indicator_matrix(
    stay_ids: list[int],
    stay_atoms: dict[int, list[tuple[int, str, bool]]],
    paths: list[tuple[str, ...]],
) -> np.ndarray:
    ind = np.zeros((len(stay_ids), len(paths)), dtype=np.float64)
    have = []
    for sid in stay_ids:
        have.append(set(enumerate_stay_paths(stay_atoms.get(int(sid), []))))
    for j, p in enumerate(paths):
        for i, sset in enumerate(have):
            if p in sset:
                ind[i, j] = 1.0
    return ind


def auprc_static_paths(
    y: np.ndarray,
    static: np.ndarray,
    path_ind: np.ndarray,
) -> float:
    if y.min() == y.max():
        return float("nan")
    if path_ind.size == 0:
        X = static
    else:
        X = np.column_stack([static, path_ind])
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    return float(average_precision_score(y, p))
