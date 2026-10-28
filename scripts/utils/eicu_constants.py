"""Frozen eICU-CRD v2.0 name maps for HA-AKI external reproduction.

Do not learn names from mortality. Do not refit MIMIC thresholds here.
Times in eICU are minutes from ICU admit; we lift them onto a dummy epoch
so existing KDIGO code (pandas timestamps) can be reused unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.utils.constants import PROJECT_DIR

EICU_EPOCH = pd.Timestamp("2000-01-01")

# Canonical output root (never writes into data/processed MIMIC tables).
EICU_OUT_DIR = Path(PROJECT_DIR) / "data" / "external" / "eicu"
MIMIC_PATHS_DIR = (
    Path(PROJECT_DIR) / "data" / "processed" / "paths" / "path_final_7var_motif84_labs11"
)

# Pre-specified urine-complete subset: at least this many measured urine hours
# in the ICU stay used for KDIGO (not tuned on eICU outcomes).
URINE_COMPLETE_MIN_HOURS = 12

# ---------------------------------------------------------------------------
# Labs: exact eICU labname → project variable. Urinary / fluid-only names omitted.
# ---------------------------------------------------------------------------
LAB_NAME_TO_VAR: dict[str, str] = {
    "creatinine": "creatinine",
    "BUN": "bun",
    "potassium": "potassium",
    "sodium": "sodium",
    "bicarbonate": "bicarbonate",
    "HCO3": "bicarbonate",
    "lactate": "lactate",
    "pH": "ph",
    "platelets x 1000": "platelet",
    "WBC x 1000": "wbc",
    "Hgb": "hemoglobin",
}

# ---------------------------------------------------------------------------
# Vitals: eICU column → project variable
# ---------------------------------------------------------------------------
PERIODIC_COL_TO_VAR: dict[str, str] = {
    "heartrate": "heart_rate",
    "sao2": "spo2",
    "respiration": "respiratory_rate",
    "systemicsystolic": "sbp",
    "systemicdiastolic": "dbp",
    "systemicmean": "mbp",
}

APERIODIC_COL_TO_VAR: dict[str, str] = {
    "noninvasivesystolic": "sbp",
    "noninvasivediastolic": "dbp",
    "noninvasivemean": "mbp",
}

# ---------------------------------------------------------------------------
# Urine: celllabel / cellpath whitelist. Chest tubes and EVD excluded.
# ---------------------------------------------------------------------------
URINE_EXACT_LABELS: frozenset[str] = frozenset({"urine"})
URINE_CONTAINS: tuple[str, ...] = (
    "urine",
    "foley",
    "urinary catheter",
    "urethral",
    "nephrostomy",
    "straight catheter",
    "condom catheter",
    "suprapubic",
)
URINE_EXCLUDE: tuple[str, ...] = (
    "chest tube",
    "evd",
    "ventriculostomy",
    "stool",
    "emesis",
    "gastric",
    "jp drain",
    "jackson",
    "ng output",
    "nasogastric",
)

# ---------------------------------------------------------------------------
# GCS (nurseCharting)
# ---------------------------------------------------------------------------
GCS_LABEL = "glasgow coma score"
GCS_TOTAL_NAMES: frozenset[str] = frozenset({"gcs total", "total"})
GCS_EYE_NAMES: frozenset[str] = frozenset({"eyes", "eye", "eye opening"})
GCS_MOTOR_NAMES: frozenset[str] = frozenset({"motor"})
GCS_VERBAL_NAMES: frozenset[str] = frozenset({"verbal"})

# ---------------------------------------------------------------------------
# Past history / ESRD (path substrings, lowercased)
# ---------------------------------------------------------------------------
ESRD_PATH_TOKENS: tuple[str, ...] = (
    "hemodialysis",
    "peritoneal dialysis",
    "renal transplant",
)
CKD_PATH_TOKENS: tuple[str, ...] = (
    "renal insufficiency",
    "renal failure",
)
COMORBID_PATH_TOKENS: dict[str, tuple[str, ...]] = {
    "CHF": ("congestive heart failure", "/chf"),
    "Diabetes": ("diabetes",),
    "HTN": ("hypertension",),
    "COPD": ("/copd", "copd "),
    "LiverDisease": ("cirrhosis",),
    "Arrhythmia": ("arrhythmia",),
}


def is_urine_cell(celllabel: str, cellpath: str = "") -> bool:
    """True iff this intakeOutput row is urine (missing ≠ 0 still applies later)."""
    label = str(celllabel or "").strip().lower()
    path = str(cellpath or "").strip().lower()
    blob = f"{label} {path}"
    if "nephrostomy" not in blob:
        if any(tok in blob for tok in URINE_EXCLUDE):
            return False
    if label in URINE_EXACT_LABELS:
        return True
    return any(tok in blob for tok in URINE_CONTAINS)


def parse_eicu_age(raw) -> float:
    """eICU stores age as string; '> 89' matches MIMIC elderly shift."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return float("nan")
    s = str(raw).strip()
    if not s:
        return float("nan")
    if s.startswith(">"):
        return 91.5
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_eicu_gender(raw) -> str:
    s = str(raw or "").strip().lower()
    if s.startswith("m"):
        return "M"
    if s.startswith("f"):
        return "F"
    return ""


def offset_to_time(offset_min) -> pd.Timestamp:
    """Lift minutes-from-unit-admit onto the dummy epoch."""
    if offset_min is None or (isinstance(offset_min, float) and pd.isna(offset_min)):
        return pd.NaT
    try:
        m = float(offset_min)
    except (TypeError, ValueError):
        return pd.NaT
    if not pd.notna(m):
        return pd.NaT
    return EICU_EPOCH + pd.Timedelta(minutes=m)


def maybe_creatinine_to_mgdl(value: float, unit: str) -> float:
    """eICU US labs are mg/dL; convert µmol/L if the unit says so."""
    u = str(unit or "").lower().replace("μ", "u").replace("µ", "u")
    if "umol" in u:
        return float(value) / 88.4
    return float(value)


def path_has_any(path: str, tokens: tuple[str, ...]) -> bool:
    blob = str(path or "").lower()
    return any(tok in blob for tok in tokens)


def gcs_component(valname: str) -> str | None:
    n = str(valname or "").strip().lower()
    if n in GCS_TOTAL_NAMES:
        return "gcs_total"
    if n in GCS_EYE_NAMES:
        return "gcs_eye"
    if n in GCS_MOTOR_NAMES:
        return "gcs_motor"
    if n in GCS_VERBAL_NAMES:
        return "gcs_verbal"
    return None
