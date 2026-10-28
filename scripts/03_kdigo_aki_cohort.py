#!/usr/bin/env python3
"""
03 — KDIGO AKI identification, onset, CA/HA, first HA stay, patient flow.

Primary analysis cohort (protocol §2.0 / mentor design):
  1) Identify AKI by creatinine **and/or** urine (KDIGO joint criteria)
  2) Keep HA-AKI stays only
  3) Per SUBJECT_ID: first HA-AKI stay (by onset time)
  4) Landmark-eligible at t0+48h

Important (fixes vs earlier bugs):
  - Selection order is HA-first, then first HA stay — NOT first-any-AKI then HA.
    (A patient with earlier CA then later HA must still be able to enter via HA.)
  - Urine-only AKI is allowed: do not require ≥2 creatinine values to evaluate a stay.
    Creatinine staging still uses available SCr; urine oliguria can define onset alone.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from scripts.utils.constants import (  # noqa: E402
    CA_MAX_HOURS,
    FEATURE_WINDOW_HOURS,
    LANDMARK_HOURS,
    MIN_SCROBS,
    OUTPUT_TABLES,
)
from scripts.utils.io_utils import (  # noqa: E402
    append_flow_step,
    load_parquet,
    save_parquet,
    write_manifest,
)
from scripts.utils.kdigo import (  # noqa: E402
    classify_ca_ha,
    combine_stages,
    creatinine_stages_for_series,
    detect_onset,
    urine_rate_ml_kg_h,
    urine_stage_from_rolling_rates,
)


def process_stay_arrays(
    stay: pd.Series,
    cr_times: np.ndarray | None,
    cr_vals: np.ndarray | None,
    urine_times: np.ndarray | None,
    urine_vals: np.ndarray | None,
    weight_kg: float,
) -> dict | None:
    """
    Return one AKI-positive stay record, or None if never stage≥1.

    Creatinine path: used when ≥1 SCr exists; reliable relative/absolute staging
    needs priors (see kdigo.baseline_creatinine_prior_only). MIN_SCROBS is a
    quality flag, not a hard exclusion for urine-defined AKI.

    Urine path: hourly oliguria staging with is_measured (missing ≠ 0).
    Onset = earliest of creatinine-combined and urine-only detections.
    """
    intime = pd.Timestamp(stay["INTIME"])
    outtime = pd.Timestamp(stay["OUTTIME"])

    has_cr = (
        cr_times is not None
        and cr_vals is not None
        and len(cr_vals) > 0
        and len(cr_times) == len(cr_vals)
    )
    has_urine = (
        urine_times is not None
        and urine_vals is not None
        and len(urine_vals) > 0
        and len(urine_times) == len(urine_vals)
    )
    if not has_cr and not has_urine:
        return None

    horizon_h = min(int(np.ceil((outtime - intime).total_seconds() / 3600.0)), 14 * 24)
    if horizon_h < 1:
        return None
    hours = np.arange(0, horizon_h + 1, dtype=float)

    # ----- Urine hourly grid (always if any urine; empty grid if none) -----
    urine_ml = np.full(len(hours), np.nan)
    is_meas = np.zeros(len(hours), dtype=bool)
    if has_urine:
        uh = (urine_times - np.datetime64(intime)) / np.timedelta64(1, "h")
        uh = np.floor(uh.astype(float)).astype(int)
        for h, v in zip(uh, urine_vals):
            if 0 <= h <= horizon_h and np.isfinite(v):
                if not is_meas[h]:
                    urine_ml[h] = float(v)
                    is_meas[h] = True
                else:
                    urine_ml[h] += float(v)

    w = weight_kg if np.isfinite(weight_kg) and weight_kg > 0 else 70.0
    rates = urine_rate_ml_kg_h(urine_ml, w)
    uo_stages_hourly = urine_stage_from_rolling_rates(rates, is_meas)
    n_urine_hours = int(is_meas.sum())

    # ----- Creatinine series (optional) -----
    n_creatinine = 0
    times_h = np.array([], dtype=float)
    scr = np.array([], dtype=float)
    cr_stages = np.array([], dtype=int)
    if has_cr:
        times_h = (cr_times - np.datetime64(intime)) / np.timedelta64(1, "h")
        times_h = times_h.astype(float)
        scr = cr_vals.astype(float)
        order = np.argsort(times_h)
        times_h = times_h[order]
        scr = scr[order]
        n_creatinine = int(len(scr))
        # Stage series even with 1 point (will stay 0 without prior); ≥MIN_SCROBS preferred
        cr_stages = creatinine_stages_for_series(times_h, scr)

    # ----- Onset candidates: Cr∪UO at Cr times, and pure UO hourly -----
    candidates: list[tuple[float, int, str]] = []

    if n_creatinine > 0:
        uo_at_cr = np.zeros(len(times_h), dtype=int)
        for i, t in enumerate(times_h):
            hi = int(np.floor(t))
            if 0 <= hi < len(uo_stages_hourly):
                uo_at_cr[i] = uo_stages_hourly[hi]
        combined = np.array(
            [
                combine_stages(int(cr_stages[i]), int(uo_at_cr[i]))
                for i in range(len(times_h))
            ]
        )
        onset_cr = detect_onset(times_h, combined)
        if onset_cr is not None:
            candidates.append((onset_cr[0], onset_cr[1], "creatinine_or_combined"))

    onset_uo = detect_onset(hours, uo_stages_hourly)
    if onset_uo is not None:
        candidates.append((onset_uo[0], onset_uo[1], "urine"))

    if not candidates:
        return None

    onset_h, onset_stage, onset_source = min(candidates, key=lambda x: x[0])
    # If both fire at same hour, prefer noting both
    sources_at_onset = {s for (h, _st, s) in candidates if abs(h - onset_h) < 1e-9}
    if len(sources_at_onset) > 1:
        onset_by = "creatinine_and_urine"
    elif "urine" in sources_at_onset and "creatinine_or_combined" not in sources_at_onset:
        onset_by = "urine_only"
    elif "creatinine_or_combined" in sources_at_onset:
        # Check whether pure urine also would have staged at that time
        hi = int(np.floor(onset_h))
        uo_st = int(uo_stages_hourly[hi]) if 0 <= hi < len(uo_stages_hourly) else 0
        cr_only = 0
        if n_creatinine > 0:
            # stage from creatinine series at nearest sample ≤ onset
            idx = np.where(times_h <= onset_h + 1e-9)[0]
            if len(idx):
                cr_only = int(cr_stages[idx[-1]])
        if uo_st >= 1 and cr_only >= 1:
            onset_by = "creatinine_and_urine"
        elif uo_st >= 1 and cr_only < 1:
            onset_by = "urine_only"
        else:
            onset_by = "creatinine_only"
    else:
        onset_by = onset_source

    onset_time = intime + pd.Timedelta(hours=float(onset_h))
    ca_ha = classify_ca_ha(onset_time, intime, CA_MAX_HOURS)

    landmark_time = onset_time + pd.Timedelta(hours=LANDMARK_HOURS)
    death_time = stay["DEATHTIME"]
    disch_time = stay["DISCHTIME"]
    died_before = False
    if pd.notna(death_time):
        died_before = pd.Timestamp(death_time) < landmark_time
    discharged_before = False
    if pd.notna(disch_time):
        discharged_before = (
            pd.Timestamp(disch_time) < landmark_time
            and int(stay["HOSPITAL_MORTALITY"]) == 0
        )

    y_primary = np.nan if died_before else int(stay["HOSPITAL_MORTALITY"])

    return {
        "SUBJECT_ID": int(stay["SUBJECT_ID"]),
        "HADM_ID": int(stay["HADM_ID"]),
        "ICUSTAY_ID": int(stay["ICUSTAY_ID"]),
        "AGE": float(stay["AGE"]),
        "GENDER": stay["GENDER"],
        "ADMISSION_TYPE": stay["ADMISSION_TYPE"],
        "INTIME": intime,
        "OUTTIME": outtime,
        "DISCHTIME": disch_time,
        "DEATHTIME": death_time,
        "HOSPITAL_MORTALITY": int(stay["HOSPITAL_MORTALITY"]),
        "onset_time": onset_time,
        "onset_hour_from_icu": float(onset_h),
        "onset_stage": int(onset_stage),
        "onset_by": onset_by,
        "ca_ha": ca_ha,
        "landmark_time": landmark_time,
        "died_before_landmark": int(died_before),
        "discharged_before_landmark": int(discharged_before),
        "y_primary": y_primary,
        "n_creatinine": n_creatinine,
        "n_urine_hours_measured": n_urine_hours,
        "scr_enough_for_cr_staging": int(n_creatinine >= MIN_SCROBS),
        "weight_kg_used": float(w),
    }


def main() -> None:
    print("=" * 60, flush=True)
    print("03: KDIGO AKI cohort + HA-first primary selection", flush=True)
    print("=" * 60, flush=True)

    base = load_parquet("icu_base", subdir="raw")
    for c in ["INTIME", "OUTTIME", "DISCHTIME", "DEATHTIME"]:
        if c in base.columns:
            base[c] = pd.to_datetime(base[c])

    print("  loading labs ...", flush=True)
    labs = load_parquet("meas_labs", subdir="raw")
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    cr = labs[labs["variable"] == "creatinine"][
        ["ICUSTAY_ID", "charttime", "value"]
    ].copy()
    del labs

    print("  loading urine ...", flush=True)
    urine = load_parquet("meas_urine", subdir="raw")
    urine["charttime"] = pd.to_datetime(urine["charttime"])
    urine = urine[["ICUSTAY_ID", "charttime", "value"]]

    print("  loading weights ...", flush=True)
    try:
        vitals = load_parquet("meas_vitals", subdir="raw")
        wdf = vitals[vitals["variable"] == "weight_kg"].groupby("ICUSTAY_ID")[
            "value"
        ].median()
        del vitals
    except FileNotFoundError:
        wdf = pd.Series(dtype=float)

    print("  building stay-indexed dicts ...", flush=True)
    cr_grp = {
        k: (g["charttime"].values.astype("datetime64[ns]"), g["value"].to_numpy(float))
        for k, g in cr.groupby("ICUSTAY_ID", sort=False)
    }
    ur_grp = {
        k: (g["charttime"].values.astype("datetime64[ns]"), g["value"].to_numpy(float))
        for k, g in urine.groupby("ICUSTAY_ID", sort=False)
    }
    del cr, urine

    flow: list = []
    append_flow_step(
        flow, "00_icu_base", base["ICUSTAY_ID"].nunique(), base["SUBJECT_ID"].nunique()
    )

    records = []
    n = len(base)
    n_skip_no_data = 0
    for i, (_, stay) in enumerate(base.iterrows()):
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"  processing stay {i+1}/{n} ...", flush=True)
        sid = int(stay["ICUSTAY_ID"])
        # Allow urine-only stays (no creatinine series required)
        if sid in cr_grp:
            ct, cv = cr_grp[sid]
        else:
            ct, cv = None, None
        if sid in ur_grp:
            ut, uv = ur_grp[sid]
        else:
            ut, uv = None, None
        if ct is None and ut is None:
            n_skip_no_data += 1
            continue
        w = float(wdf[sid]) if sid in wdf.index else 70.0
        rec = process_stay_arrays(stay, ct, cv, ut, uv, w)
        if rec is not None:
            records.append(rec)

    if not records:
        raise RuntimeError("No AKI stays detected — check labs/urine extraction.")

    aki = pd.DataFrame(records)
    append_flow_step(
        flow,
        "01_kdigo_aki_any",
        aki["ICUSTAY_ID"].nunique(),
        aki["SUBJECT_ID"].nunique(),
        "stage>=1 by creatinine and/or urine (urine-only allowed)",
    )
    if "onset_by" in aki.columns:
        print("  onset_by counts:", aki["onset_by"].value_counts().to_dict(), flush=True)

    # --- Mentor / protocol order: HA-AKI first, then first HA stay per subject ---
    # (Do NOT take first-any-AKI then filter HA: that drops later HA after earlier CA.)
    aki = aki.sort_values(["SUBJECT_ID", "onset_time"])
    # Diagnostic only: first AKI of any type (CA or HA)
    aki["is_first_aki_any"] = ~aki.duplicated(subset=["SUBJECT_ID"], keep="first")

    ha_all = aki[aki["ca_ha"] == "HA"].copy()
    append_flow_step(
        flow,
        "02_ha_aki_stays",
        ha_all["ICUSTAY_ID"].nunique(),
        ha_all["SUBJECT_ID"].nunique(),
        f"all HA-AKI stays (CA_MAX_HOURS={CA_MAX_HOURS}); before first-HA dedupe",
    )

    ha_all = ha_all.sort_values(["SUBJECT_ID", "onset_time"])
    ha_all["is_first_ha_aki_stay"] = ~ha_all.duplicated(
        subset=["SUBJECT_ID"], keep="first"
    )
    first_ha = ha_all[ha_all["is_first_ha_aki_stay"]].copy()
    append_flow_step(
        flow,
        "03_first_ha_aki_per_subject",
        first_ha["ICUSTAY_ID"].nunique(),
        first_ha["SUBJECT_ID"].nunique(),
        "per SUBJECT_ID: earliest HA-AKI onset stay",
    )

    # Merge first-HA flag back to full aki table for audit
    first_ha_ids = set(first_ha["ICUSTAY_ID"].tolist())
    aki["is_first_ha_aki_stay"] = aki["ICUSTAY_ID"].isin(first_ha_ids)
    # Keep legacy column name for downstream readers, but semantics = first HA
    aki["is_first_aki_stay"] = aki["is_first_ha_aki_stay"]

    primary = first_ha[
        (first_ha["died_before_landmark"] == 0)
        & (first_ha["discharged_before_landmark"] == 0)
    ].copy()
    primary["primary_eligible"] = 1
    append_flow_step(
        flow,
        "04_landmark_eligible_48h",
        primary["ICUSTAY_ID"].nunique(),
        primary["SUBJECT_ID"].nunique(),
        f"alive & hospitalized at t0+{LANDMARK_HOURS}h",
    )

    aki["primary_eligible"] = aki["ICUSTAY_ID"].isin(primary["ICUSTAY_ID"]).astype(int)

    save_parquet(aki, "aki_cohort_kdigo", subdir="processed")
    save_parquet(primary, "aki_primary_ha48", subdir="processed")

    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    flow_path = os.path.join(OUTPUT_TABLES, "patient_flow_03.csv")
    pd.DataFrame(flow).to_csv(flow_path, index=False)

    print("\n  --- Flow ---", flush=True)
    print(pd.DataFrame(flow).to_string(index=False), flush=True)
    print(f"  stays skipped (no Cr and no urine): {n_skip_no_data}", flush=True)
    if len(primary):
        print(
            f"\n  Primary N={len(primary)} "
            f"mortality={primary['HOSPITAL_MORTALITY'].mean()*100:.1f}%",
            flush=True,
        )
        if "onset_by" in primary.columns:
            print(
                "  primary onset_by:",
                primary["onset_by"].value_counts().to_dict(),
                flush=True,
            )

    write_manifest(
        {
            "stage": "03_kdigo_aki_cohort",
            "design": {
                "population": "HA-AKI",
                "selection_order": "HA_first_then_first_HA_stay_per_subject",
                "aki_criteria": "KDIGO_creatinine_and_or_urine",
                "urine_only_aki_allowed": True,
                "min_scr_obs_for_flag": MIN_SCROBS,
                "t0": "onset",
                "feature_window_h": FEATURE_WINDOW_HOURS,
                "landmark_h": LANDMARK_HOURS,
            },
            "n_aki_stays": int(len(aki)),
            "n_ha_stays": int(len(ha_all)),
            "n_first_ha": int(len(first_ha)),
            "n_primary": int(len(primary)),
            "primary_mortality": float(primary["HOSPITAL_MORTALITY"].mean())
            if len(primary)
            else None,
            "flow": flow_path,
        }
    )
    print("\n03 done.", flush=True)


if __name__ == "__main__":
    main()
