"""eICU name maps and urine/ESRD rules. No eICU files required."""

from __future__ import annotations

import pandas as pd

from scripts.utils.eicu_constants import (
    EICU_EPOCH,
    LAB_NAME_TO_VAR,
    URINE_COMPLETE_MIN_HOURS,
    gcs_component,
    is_urine_cell,
    maybe_creatinine_to_mgdl,
    offset_to_time,
    parse_eicu_age,
    parse_eicu_gender,
    path_has_any,
    ESRD_PATH_TOKENS,
)


def test_lab_names_do_not_include_urinary_creatinine():
    assert "creatinine" in LAB_NAME_TO_VAR
    assert "urinary creatinine" not in LAB_NAME_TO_VAR
    assert "urinary sodium" not in LAB_NAME_TO_VAR
    assert LAB_NAME_TO_VAR["pH"] == "ph"
    assert LAB_NAME_TO_VAR["HCO3"] == "bicarbonate"
    assert LAB_NAME_TO_VAR["platelets x 1000"] == "platelet"


def test_urine_whitelist_and_chest_tube_excluded():
    assert is_urine_cell("Urine", "Flowsheet|Urine")
    assert is_urine_cell("Output (mL)-Urinary Catheter Indwelling", "")
    assert is_urine_cell("Output (ml)-Nephrostomy Left", "")
    assert not is_urine_cell("Output (ml)-Chest Tube #1 Right", "")
    assert not is_urine_cell("EVD Output (ml)-ICP/External Ventricular Drain", "")


def test_age_gender_offset():
    assert parse_eicu_age("> 89") == 91.5
    assert parse_eicu_age("67") == 67.0
    assert parse_eicu_gender("Female") == "F"
    assert parse_eicu_gender("Male") == "M"
    t = offset_to_time(120)
    assert t == EICU_EPOCH + pd.Timedelta(minutes=120)
    assert pd.isna(offset_to_time(None))


def test_creatinine_unit_and_gcs():
    assert abs(maybe_creatinine_to_mgdl(88.4, "umol/L") - 1.0) < 1e-6
    assert maybe_creatinine_to_mgdl(1.2, "mg/dL") == 1.2
    assert gcs_component("GCS Total") == "gcs_total"
    assert gcs_component("Eyes") == "gcs_eye"
    assert gcs_component("pain score") is None


def test_esrd_tokens_and_urine_complete_pre_specified():
    assert path_has_any(
        "notes/.../Renal Failure/renal failure - hemodialysis", ESRD_PATH_TOKENS
    )
    assert not path_has_any(
        "notes/.../Renal Insufficiency/renal insufficiency - creatinine 1-2",
        ESRD_PATH_TOKENS,
    )
    assert URINE_COMPLETE_MIN_HOURS == 12


def test_eicu_outputs_do_not_share_mimic_processed_dir():
    from scripts.utils.constants import DATA_PROCESSED
    from scripts.utils.eicu_constants import EICU_OUT_DIR

    assert "external" in str(EICU_OUT_DIR).replace("\\", "/")
    assert Path_as_posix(EICU_OUT_DIR) != Path_as_posix(DATA_PROCESSED)


def Path_as_posix(p) -> str:
    from pathlib import Path

    return Path(p).resolve().as_posix()
