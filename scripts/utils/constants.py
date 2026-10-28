"""
AKI Research Project — Constants & Configuration
=================================================
Paths, MIMIC ITEMIDs, cohort rules, study design (HA-AKI 48h landmark).
"""

from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# Paths (repo-relative: works on any machine/OS)
# =============================================================================
PROJECT_DIR = str(Path(__file__).resolve().parent.parent.parent)
# Local copy of essential MIMIC-III tables (copied into the repo)
MIMIC_DIR = os.path.join(PROJECT_DIR, "data", "mimic-iii")
# Fallback external extract (only used if it exists on this machine)
MIMIC_DIR_FALLBACK = os.path.join(PROJECT_DIR, "data", "mimic-iii-external")
# eICU-CRD v2.0 (external reproduction only — never fit thresholds here)
EICU_DIR = os.path.join(PROJECT_DIR, "data", "eicu-crd")
DATA_EXTERNAL = os.path.join(PROJECT_DIR, "data", "external")

DATA_RAW = os.path.join(PROJECT_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_DIR, "data", "processed")
DATA_SPLITS = os.path.join(DATA_PROCESSED, "splits")
OUTPUT_FIGURES = os.path.join(PROJECT_DIR, "outputs", "figures")
OUTPUT_TABLES = os.path.join(PROJECT_DIR, "outputs", "tables")
OUTPUT_REPORTS = os.path.join(PROJECT_DIR, "outputs", "reports")
DOCS_DIR = os.path.join(PROJECT_DIR, "docs")

# =============================================================================
# Study design (locked) — see docs/10 protocol §2.0
# =============================================================================
CA_MAX_HOURS = 0.0              # onset > intime + this → HA
FEATURE_WINDOW_HOURS = 48       # primary observation window after onset
FEATURE_WINDOW_HOURS_S1 = 72    # secondary window
LANDMARK_HOURS = 48             # prediction made at t0 + this
PRIMARY_POPULATION = "HA"       # HA only for primary
FIRST_AKI_STAY_ONLY = True
EXCLUDE_DIED_BEFORE_LANDMARK = True
EXCLUDE_DISCHARGED_BEFORE_LANDMARK = True

# =============================================================================
# Cohort Definition
# =============================================================================
MIN_AGE = 18
MAX_AGE = 120
MIN_ICU_LOS_HOURS = 24          # base ICU filter (before HA/48h rules)
# Quality flag for creatinine staging (needs priors). NOT a hard exclusion:
# urine-only KDIGO AKI is allowed even if n_creatinine < MIN_SCROBS.
MIN_SCROBS = 2

# =============================================================================
# AKI ICD-9 (sensitivity / comorbidity only — NOT primary cohort definition)
# =============================================================================
AKI_ICD9_CODES = {
    "5845": "ATN",
    "5846": "cortical necrosis",
    "5847": "medullary necrosis",
    "5848": "other specified",
    "5849": "unspecified",
}

CKD_ICD9_PREFIXES = ("585",)
ESRD_ICD9_CODES = frozenset({"5856", "V4511", "V4512", "V560", "V561", "V568"})

# =============================================================================
# Lab ITEMIDs (LABEVENTS / D_LABITEMS)
# =============================================================================
LAB_ITEMIDS = {
    "creatinine": 50912,
    "bun": 51006,
    "potassium": 50971,
    "sodium": 50983,
    "bicarbonate": 50882,
    "chloride": 50902,
    "calcium": 50893,
    "phosphate": 50970,
    "magnesium": 50960,
    "ph": 50820,
    "pco2": 50818,
    "po2": 50821,
    "hematocrit": 51221,
    "hemoglobin": 51222,
    "platelet": 51265,
    "wbc": 51301,          # also 51300 in some extracts
    "wbc_alt": 51300,
    "glucose": 50931,
    "albumin": 50862,
    "lactate": 50813,
    "bilirubin_total": 50885,
    "alt": 50861,
    "ast": 50878,
    "alp": 50863,
    "anion_gap": 50868,
    "rdw": 51277,
    "inr": 51237,
    "ptt": 51275,
}

# Reverse map itemid -> name (primary names only)
LAB_ITEMID_TO_NAME = {
    50912: "creatinine",
    51006: "bun",
    50971: "potassium",
    50983: "sodium",
    50882: "bicarbonate",
    50902: "chloride",
    50893: "calcium",
    50970: "phosphate",
    50960: "magnesium",
    50820: "ph",
    50818: "pco2",
    50821: "po2",
    51221: "hematocrit",
    51222: "hemoglobin",
    51265: "platelet",
    51301: "wbc",
    51300: "wbc",
    50931: "glucose",
    50862: "albumin",
    50813: "lactate",
    50885: "bilirubin_total",
    50861: "alt",
    50878: "ast",
    50863: "alp",
    50868: "anion_gap",
    51277: "rdw",
    51237: "inr",
    51275: "ptt",
}

# =============================================================================
# Vital ITEMIDs (CHARTEVENTS) — Metavision + CareVue
# =============================================================================
# name -> list of itemids (any match maps to that name)
VITAL_ITEMID_GROUPS = {
    "heart_rate": [220045, 211],
    "sbp": [220179, 220050, 455, 51],          # NIBP sys + ABP sys
    "dbp": [220180, 220051, 8441, 8368],
    "mbp": [220181, 220052, 456, 52],
    "respiratory_rate": [220210, 618, 3603],
    "spo2": [220277, 646, 6702],
    "temperature_c": [223761, 676],
    "temperature_f": [223762, 678],
    "fio2": [223835, 3420, 3422, 190],
    "gcs_eye": [220739, 184],
    "gcs_motor": [223901, 454],
    "gcs_verbal": [223900, 723],
    "weight_kg": [226512, 224639],
    "weight_lb": [226531],
}

VITAL_ITEMID_TO_NAME = {
    iid: name for name, ids in VITAL_ITEMID_GROUPS.items() for iid in ids
}

# =============================================================================
# Urine ITEMIDs (OUTPUTEVENTS)
# =============================================================================
URINE_ITEMIDS = {
    # CareVue
    40055, 40056, 40057, 40058, 40059, 40060, 40061, 40062, 40063, 40064,
    40065, 40066, 40067, 40068, 40069,
    # Metavision
    226559, 226560, 226561, 226563, 226564, 226565, 226566, 226567,
    226584, 227488,
}

# =============================================================================
# Medication keyword categories (PRESCRIPTIONS.DRUG, case-insensitive)
# =============================================================================
MED_CATEGORIES = {
    "vasopressor": [
        "norepinephrine", "epinephrine", "phenylephrine", "vasopressin",
        "dopamine", "dobutamine",
    ],
    "diuretic": [
        "furosemide", "bumetanide", "torsemide", "mannitol",
        "spironolactone", "hydrochlorothiazide", "chlorothiazide",
    ],
    "nephrotoxin": [
        "vancomycin", "gentamicin", "tobramycin", "amikacin",
        "ibuprofen", "ketorolac", "naproxen",
    ],
}

# =============================================================================
# Physiological ranges (outlier clipping)
# =============================================================================
PHYSIO_RANGES = {
    "creatinine": (0.2, 20.0),
    "bun": (2.0, 200.0),
    "potassium": (2.0, 9.0),
    "sodium": (110.0, 170.0),
    "bicarbonate": (5.0, 50.0),
    "ph": (6.8, 7.8),
    "hematocrit": (10.0, 60.0),
    "hemoglobin": (3.0, 20.0),
    "platelet": (10.0, 1500.0),
    "wbc": (0.1, 500.0),
    "glucose": (20.0, 1000.0),
    "lactate": (0.1, 30.0),
    "inr": (0.5, 20.0),
    "ptt": (15.0, 200.0),
    "heart_rate": (20, 250),
    "sbp": (40, 280),
    "dbp": (20, 180),
    "mbp": (30, 220),
    "respiratory_rate": (4, 60),
    "temperature_c": (28.0, 44.0),
    "temperature_f": (85.0, 110.0),
    "spo2": (50.0, 100.0),
    "fio2": (21.0, 100.0),
    "gcs_eye": (1, 4),
    "gcs_motor": (1, 6),
    "gcs_verbal": (1, 5),
    "urine_output_ml": (0.0, 5000.0),
    "weight_kg": (20.0, 300.0),
}

# =============================================================================
# KDIGO
# =============================================================================
KDIGO_RATIO_1 = 1.5
KDIGO_RATIO_2 = 2.0
KDIGO_RATIO_3 = 3.0
KDIGO_ABS_INCREASE = 0.3  # mg/dL within 48h
KDIGO_CR_MAX = 4.0
BASELINE_LOOKBACK_HOURS = 168.0   # 7 days prior for relative baseline
ABS_INCREASE_WINDOW_HOURS = 48.0
URINE_STAGE1_ML_KG_H = 0.5
URINE_STAGE1_HOURS = 6
URINE_STAGE2_HOURS = 12
URINE_STAGE3_ML_KG_H = 0.3
URINE_STAGE3_HOURS = 24

# =============================================================================
# Variable tiers (clinical pools — audit may demote)
# =============================================================================
TIER_A_SHAPELET = (
    "creatinine", "bun", "urine_output_ml",
    "heart_rate", "sbp", "dbp", "mbp",
    "respiratory_rate", "spo2",
    "gcs_eye", "gcs_motor", "gcs_verbal", "gcs_total",
    "sodium", "potassium", "bicarbonate",
    "platelet", "wbc", "hemoglobin",
    "lactate", "ph",
)

TIER_B_SPARSE = (
    "albumin", "alp", "alt", "ast", "bilirubin_total",
    "anion_gap", "chloride", "calcium", "magnesium", "phosphate",
    "hematocrit", "rdw", "inr", "ptt", "glucose", "fio2",
)

TIER_C_STATIC = (
    "AGE", "GENDER", "ADMISSION_TYPE",
    "CHF", "Arrhythmia", "HTN", "Diabetes", "LiverDisease", "COPD",
    "CKD_FLAG", "ESRD_FLAG",
)

TIER_A_MIN_PATIENT_COVERAGE = 0.70
TIER_A_MAX_TIMEPOINT_MISSING = 0.60

# =============================================================================
# Split
# =============================================================================
RANDOM_SEED = 42
SPLIT_RATIOS = (0.60, 0.20, 0.20)  # train, val, test
SPLIT_BY = "SUBJECT_ID"

# =============================================================================
# Elixhauser (simplified ICD9 → category)
# =============================================================================
ELIXHAUSER_ICD9_MAP = {
    "39891": "CHF", "40201": "CHF", "40211": "CHF", "40291": "CHF",
    "40401": "CHF", "40403": "CHF", "40411": "CHF", "40413": "CHF",
    "40491": "CHF", "40493": "CHF", "4280": "CHF", "4281": "CHF",
    "42820": "CHF", "42821": "CHF", "42822": "CHF", "42823": "CHF",
    "42830": "CHF", "42831": "CHF", "42832": "CHF", "42833": "CHF",
    "42840": "CHF", "42841": "CHF", "42842": "CHF", "42843": "CHF",
    "4289": "CHF",
    "4260": "Arrhythmia", "42613": "Arrhythmia", "4267": "Arrhythmia",
    "4269": "Arrhythmia", "4270": "Arrhythmia", "4271": "Arrhythmia",
    "4272": "Arrhythmia", "42731": "Arrhythmia", "42732": "Arrhythmia",
    "42760": "Arrhythmia", "4279": "Arrhythmia", "7850": "Arrhythmia",
    "4011": "HTN", "4019": "HTN", "40501": "HTN", "40509": "HTN",
    "40511": "HTN", "40519": "HTN", "40591": "HTN", "40599": "HTN",
    "25000": "Diabetes", "25001": "Diabetes", "25002": "Diabetes",
    "25003": "Diabetes", "25010": "Diabetes", "25011": "Diabetes",
    "25012": "Diabetes", "25013": "Diabetes", "25020": "Diabetes",
    "25021": "Diabetes", "25022": "Diabetes", "25023": "Diabetes",
    "25030": "Diabetes", "25031": "Diabetes", "25032": "Diabetes",
    "25033": "Diabetes",
    "5712": "LiverDisease", "5715": "LiverDisease", "5716": "LiverDisease",
    "5722": "LiverDisease", "5723": "LiverDisease", "5724": "LiverDisease",
    "5728": "LiverDisease", "4560": "LiverDisease", "4561": "LiverDisease",
    "45620": "LiverDisease", "45621": "LiverDisease",
    "490": "COPD", "4910": "COPD", "4911": "COPD", "49120": "COPD",
    "49121": "COPD", "49122": "COPD", "4918": "COPD", "4919": "COPD",
    "4920": "COPD", "4928": "COPD", "494": "COPD", "4940": "COPD",
    "4941": "COPD", "496": "COPD",
}
