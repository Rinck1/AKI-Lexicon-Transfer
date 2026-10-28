#!/usr/bin/env python3
"""eICU KDIGO HA-AKI landmark cohort. Reuses scripts/03 process_stay_arrays.

Pre-specified urine-complete flag: n_urine_hours_measured >= URINE_COMPLETE_MIN_HOURS.
Does not change onset rules.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.constants import CA_MAX_HOURS, LANDMARK_HOURS, OUTPUT_TABLES  # noqa: E402
from scripts.utils.eicu_constants import URINE_COMPLETE_MIN_HOURS  # noqa: E402
from scripts.utils.eicu_io import load_eicu, out_dir, save_eicu  # noqa: E402
from scripts.utils.io_utils import append_flow_step  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "kdigo_cohort_03", PROJ / "scripts" / "03_kdigo_aki_cohort.py"
)
_mod03 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod03)
process_stay_arrays = _mod03.process_stay_arrays


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    base = load_eicu("icu_base")
    for c in ["INTIME", "OUTTIME", "DISCHTIME", "DEATHTIME"]:
        if c in base.columns:
            base[c] = pd.to_datetime(base[c])
    _log("[42] loading labs/urine …")
    labs = load_eicu("meas_labs")
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    cr = labs.loc[labs["variable"] == "creatinine", ["ICUSTAY_ID", "charttime", "value"]]
    del labs
    urine = load_eicu("meas_urine")
    urine["charttime"] = pd.to_datetime(urine["charttime"])
    urine = urine[["ICUSTAY_ID", "charttime", "value"]]

    wdf = base.set_index("ICUSTAY_ID")["weight_kg"] if "weight_kg" in base.columns else pd.Series(dtype=float)

    cr_grp = {
        int(k): (g["charttime"].values.astype("datetime64[ns]"), g["value"].to_numpy(float))
        for k, g in cr.groupby("ICUSTAY_ID", sort=False)
    }
    ur_grp = {
        int(k): (g["charttime"].values.astype("datetime64[ns]"), g["value"].to_numpy(float))
        for k, g in urine.groupby("ICUSTAY_ID", sort=False)
    }
    del cr, urine

    flow: list = []
    append_flow_step(flow, "00_icu_base", base["ICUSTAY_ID"].nunique(), base["SUBJECT_ID"].nunique())

    records = []
    n = len(base)
    n_skip = 0
    for i, (_, stay) in enumerate(base.iterrows()):
        if (i + 1) % 2000 == 0 or i == 0:
            _log(f"  stay {i+1}/{n}")
        sid = int(stay["ICUSTAY_ID"])
        ct, cv = cr_grp.get(sid, (None, None))
        ut, uv = ur_grp.get(sid, (None, None))
        if ct is None and ut is None:
            n_skip += 1
            continue
        w = float(wdf.loc[sid]) if sid in wdf.index and pd.notna(wdf.loc[sid]) else 70.0
        rec = process_stay_arrays(stay, ct, cv, ut, uv, w)
        if rec is not None:
            records.append(rec)
    if not records:
        raise RuntimeError("No AKI stays on eICU — check lab/urine maps.")

    aki = pd.DataFrame(records)
    append_flow_step(
        flow,
        "01_kdigo_aki_any",
        aki["ICUSTAY_ID"].nunique(),
        aki["SUBJECT_ID"].nunique(),
        "stage>=1 creatinine and/or urine",
    )
    _log(f"  onset_by {aki['onset_by'].value_counts().to_dict()}")

    aki = aki.sort_values(["SUBJECT_ID", "onset_time"])
    aki["is_first_aki_any"] = ~aki.duplicated(subset=["SUBJECT_ID"], keep="first")
    ha_all = aki[aki["ca_ha"] == "HA"].copy()
    append_flow_step(
        flow,
        "02_ha_aki_stays",
        ha_all["ICUSTAY_ID"].nunique(),
        ha_all["SUBJECT_ID"].nunique(),
        f"CA_MAX_HOURS={CA_MAX_HOURS}",
    )
    ha_all = ha_all.sort_values(["SUBJECT_ID", "onset_time"])
    ha_all["is_first_ha_aki_stay"] = ~ha_all.duplicated(subset=["SUBJECT_ID"], keep="first")
    first_ha = ha_all[ha_all["is_first_ha_aki_stay"]].copy()
    append_flow_step(
        flow,
        "03_first_ha_aki_per_subject",
        first_ha["ICUSTAY_ID"].nunique(),
        first_ha["SUBJECT_ID"].nunique(),
        "earliest HA-AKI onset stay",
    )
    aki["is_first_ha_aki_stay"] = aki["ICUSTAY_ID"].isin(set(first_ha["ICUSTAY_ID"]))
    aki["is_first_aki_stay"] = aki["is_first_ha_aki_stay"]

    primary = first_ha[
        (first_ha["died_before_landmark"] == 0) & (first_ha["discharged_before_landmark"] == 0)
    ].copy()
    primary["primary_eligible"] = 1
    primary["urine_complete"] = (
        primary["n_urine_hours_measured"] >= URINE_COMPLETE_MIN_HOURS
    ).astype(int)
    append_flow_step(
        flow,
        "04_landmark_eligible_48h",
        primary["ICUSTAY_ID"].nunique(),
        primary["SUBJECT_ID"].nunique(),
        f"alive & in hospital at t0+{LANDMARK_HOURS}h",
    )
    append_flow_step(
        flow,
        "05_urine_complete_subset",
        int(primary["urine_complete"].sum()),
        primary.loc[primary["urine_complete"] == 1, "SUBJECT_ID"].nunique(),
        f"n_urine_hours_measured >= {URINE_COMPLETE_MIN_HOURS}",
    )
    aki["primary_eligible"] = aki["ICUSTAY_ID"].isin(primary["ICUSTAY_ID"]).astype(int)

    save_eicu(aki, "aki_cohort_kdigo")
    save_eicu(primary, "aki_primary_ha48")
    flow_csv = Path(OUTPUT_TABLES) / "eicu_patient_flow_42.csv"
    flow_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flow).to_csv(flow_csv, index=False)
    (out_dir() / "stage42_manifest.json").write_text(
        json.dumps(
            {
                "stage": "42_eicu_kdigo_cohort",
                "n_aki": int(len(aki)),
                "n_primary": int(len(primary)),
                "n_urine_complete": int(primary["urine_complete"].sum()),
                "mortality_primary": float(primary["HOSPITAL_MORTALITY"].mean()) if len(primary) else None,
                "onset_by": aki["onset_by"].value_counts().to_dict(),
                "n_skip_no_cr_urine": int(n_skip),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(pd.DataFrame(flow).to_string(index=False))
    if len(primary):
        _log(
            f"[42] primary N={len(primary)} "
            f"mort={primary['HOSPITAL_MORTALITY'].mean()*100:.1f}% "
            f"urine_complete={int(primary['urine_complete'].sum())}"
        )


if __name__ == "__main__":
    main()
