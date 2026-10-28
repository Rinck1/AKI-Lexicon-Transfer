#!/usr/bin/env python3
"""External-cohort path reproduction stub (docs/20 §5).

Does not exist to retune MIMIC rules. Requires a pre-built atoms table
produced with the *frozen* MIMIC train thresholds + family map.

Expected layout (create when eICU / MIMIC-IV is ready):

  data/external/<cohort>/
    stay_features.parquet      # must include HOSPITAL_MORTALITY + SHAPELET_STATIC_COLS
    atoms.parquet              # motif+lab atoms, split != test from MIMIC
    freeze_sha256.txt          # must match path_freeze_manifest.json

Then:

  python -u scripts/run_eicu_external.py
  python -u scripts/28_external_paths_stub.py --cohort-dir data/external/eicu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from scripts.utils.path_protocol import refuse_test_path  # noqa: E402

MIMIC_PATHS = PROJ / "data/processed/paths/path_final_7var_motif84_labs11"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-dir", required=True)
    p.add_argument("--paths-dir", default=str(MIMIC_PATHS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.paths_dir)
    cohort = Path(args.cohort_dir)
    refuse_test_path(src)
    freeze_p = src / "path_freeze_manifest.json"
    if not freeze_p.is_file():
        raise SystemExit("missing path_freeze_manifest.json — run 26 first")
    freeze = json.loads(freeze_p.read_text(encoding="utf-8"))
    need = [
        cohort / "stay_features.parquet",
        cohort / "atoms.parquet",
        cohort / "freeze_sha256.txt",
    ]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        raise SystemExit(
            "external cohort not ready. missing:\n  "
            + "\n  ".join(missing)
            + "\nRun: python -u scripts/run_eicu_external.py\n"
            "Atoms must use frozen MIMIC lab_thresholds.json + family map; "
            "do not refit thresholds on the external cohort."
        )
    got = (cohort / "freeze_sha256.txt").read_text(encoding="utf-8").strip()
    if got != freeze["freeze_sha256"]:
        raise SystemExit(f"freeze hash mismatch: cohort={got} mimic={freeze['freeze_sha256']}")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eicu46", PROJ / "scripts" / "46_eicu_reproduce_paths.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
