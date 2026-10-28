#!/usr/bin/env python3
"""Run the eICU external reproduction pipeline (docs/24). Never fits on eICU.

  python -u scripts/run_eicu_external.py
  python -u scripts/run_eicu_external.py --from-stage 40 --to-stage 42
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

STAGES = (40, 41, 42, 43, 44, 45, 46)
SCRIPTS = {
    40: "40_eicu_build_icu_base.py",
    41: "41_eicu_extract_measurements.py",
    42: "42_eicu_kdigo_cohort.py",
    43: "43_eicu_align_time_grid.py",
    44: "44_eicu_stay_features.py",
    45: "45_eicu_frozen_atoms.py",
    46: "46_eicu_reproduce_paths.py",
}


def _run(stage: int) -> None:
    path = PROJ / "scripts" / SCRIPTS[stage]
    spec = importlib.util.spec_from_file_location(f"eicu_stage_{stage}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    print(f"\n===== stage {stage} {SCRIPTS[stage]} =====", flush=True)
    mod.main()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-stage", type=int, default=40, choices=STAGES)
    p.add_argument("--to-stage", type=int, default=46, choices=STAGES)
    args = p.parse_args()
    if args.from_stage > args.to_stage:
        raise SystemExit("--from-stage must be <= --to-stage")
    for st in STAGES:
        if args.from_stage <= st <= args.to_stage:
            _run(st)
    print("\n[eicu] requested stages done. Outputs in data/external/eicu/", flush=True)


if __name__ == "__main__":
    main()
