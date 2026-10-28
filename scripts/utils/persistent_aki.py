"""48h landmark persistent vs resolving AKI (clinical contrast, not path mining).

Creatinine rule follows Bhatraju et al. Crit Care 2016: resolving if the last
SCr in the window fell by >=0.3 mg/dL or >=25% from the window maximum.
Urine rule: KDIGO oliguria still present in the last 6 h of the 48 h grid
(missing hours do not count as anuria).

Persistent = non-resolving creatinine OR oliguria at the window end.
Definitions use labs/urine only; they do not use death or path labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.utils.constants import KDIGO_ABS_INCREASE
from scripts.utils.kdigo import urine_rate_ml_kg_h, urine_stage_from_rolling_rates

LAST_CR_HOURS = (36, 48)  # [36, 48)
OLIGO_HOURS = (42, 48)
N_HOURS = 48


def creatinine_resolving(max_cr: float, last_cr: float) -> float:
    """1 = resolving, 0 = non-resolving, nan = cannot score."""
    if not np.isfinite(max_cr) or not np.isfinite(last_cr) or max_cr <= 0:
        return float("nan")
    drop_abs = last_cr <= max_cr - float(KDIGO_ABS_INCREASE)
    drop_rel = last_cr <= 0.75 * max_cr
    return 1.0 if (drop_abs or drop_rel) else 0.0


def _series_48(sub: pd.DataFrame, value_col: str, meas_col: str) -> tuple[np.ndarray, np.ndarray]:
    val = np.full(N_HOURS, np.nan, dtype=float)
    meas = np.zeros(N_HOURS, dtype=bool)
    hour = sub["HOUR"].to_numpy()
    v = sub[value_col].to_numpy()
    m = sub[meas_col].to_numpy() if meas_col in sub.columns else np.isfinite(v)
    for h, vv, mm in zip(hour, v, m):
        hi = int(h)
        if 0 <= hi < N_HOURS:
            val[hi] = float(vv) if np.isfinite(vv) else np.nan
            meas[hi] = bool(mm) and np.isfinite(vv)
    return val, meas


def persistent_flags_for_stay(
    sub: pd.DataFrame,
    weight_kg: float,
) -> dict:
    cr, cr_m = _series_48(sub, "creatinine", "creatinine_is_measured")
    uo, uo_m = _series_48(sub, "urine_output_ml", "urine_output_ml_is_measured")
    measured_cr = cr[cr_m]
    max_cr = float(np.nanmax(measured_cr)) if measured_cr.size else float("nan")
    last_slice = cr[LAST_CR_HOURS[0] : LAST_CR_HOURS[1]]
    last_m = cr_m[LAST_CR_HOURS[0] : LAST_CR_HOURS[1]]
    if last_m.any():
        last_cr = float(last_slice[last_m][-1])
    elif cr_m.any():
        last_cr = float(cr[cr_m][-1])
    else:
        last_cr = float("nan")
    cr_res = creatinine_resolving(max_cr, last_cr)
    rates = urine_rate_ml_kg_h(uo, float(weight_kg) if np.isfinite(weight_kg) else float("nan"))
    u_stage = urine_stage_from_rolling_rates(rates, uo_m)
    oligo_end = bool(np.max(u_stage[OLIGO_HOURS[0] : OLIGO_HOURS[1]]) >= 1)
    cr_nonres = (cr_res == 0.0)
    persistent = bool(cr_nonres or oligo_end)
    unknown = (not np.isfinite(cr_res)) and (not uo_m[OLIGO_HOURS[0] : OLIGO_HOURS[1]].any())
    return {
        "max_creatinine": max_cr,
        "last_creatinine": last_cr,
        "cr_resolving": cr_res,
        "cr_nonresolving": float(cr_nonres) if np.isfinite(cr_res) else float("nan"),
        "oliguria_end": float(oligo_end),
        "persistent_aki": float(persistent),
        "persistent_unknown": float(unknown),
    }


def build_persistent_table(
    grid: pd.DataFrame,
    cohort: pd.DataFrame,
    stay_ids: list[int],
) -> pd.DataFrame:
    wmap = cohort.set_index("ICUSTAY_ID")["weight_kg_used"].to_dict()
    g = grid[grid["ICUSTAY_ID"].isin(stay_ids)]
    rows = []
    for sid, sub in g.groupby("ICUSTAY_ID", sort=False):
        flags = persistent_flags_for_stay(sub, float(wmap.get(int(sid), np.nan)))
        flags["ICUSTAY_ID"] = int(sid)
        rows.append(flags)
    out = pd.DataFrame(rows)
    missing = sorted(set(int(s) for s in stay_ids) - set(out["ICUSTAY_ID"].astype(int)))
    if missing:
        extra = pd.DataFrame(
            {
                "ICUSTAY_ID": missing,
                "max_creatinine": np.nan,
                "last_creatinine": np.nan,
                "cr_resolving": np.nan,
                "cr_nonresolving": np.nan,
                "oliguria_end": 0.0,
                "persistent_aki": 0.0,
                "persistent_unknown": 1.0,
            }
        )
        out = pd.concat([out, extra], ignore_index=True)
    return out.sort_values("ICUSTAY_ID").reset_index(drop=True)
