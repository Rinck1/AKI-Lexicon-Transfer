#!/usr/bin/env python3
"""Run pipeline stages 01→09 sequentially (09 is optional DL baseline)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = [
    "01_build_icu_base.py",
    "02_extract_measurements.py",
    "03_kdigo_aki_cohort.py",
    "04_align_time_grid.py",
    "05_split_subjects.py",
    "06_features_post_split.py",
    "07_variable_audit.py",
    "08_baseline_early_risk.py",  # HistGBM tree baseline
    "09_dl_baseline.py",  # MLP / tabular Transformer
]


def main() -> None:
    p = argparse.ArgumentParser(description="AKI research pipeline runner")
    p.add_argument("--from-stage", type=int, default=1, help="1-9")
    p.add_argument("--to-stage", type=int, default=8, help="1-9 (default 8 skips DL)")
    p.add_argument(
        "--dl-model",
        choices=["mlp", "transformer"],
        default="mlp",
        help="Only used when stage 09 is run",
    )
    args = p.parse_args()
    if not (1 <= args.from_stage <= 9 and 1 <= args.to_stage <= 9):
        p.error("stages must be in 1..9")
    if args.from_stage > args.to_stage:
        p.error("--from-stage cannot exceed --to-stage")

    scripts_dir = Path(__file__).resolve().parent
    proj = scripts_dir.parent
    for i in range(args.from_stage, args.to_stage + 1):
        script = scripts_dir / STAGES[i - 1]
        cmd = [sys.executable, str(script)]
        if i == 9:
            cmd.extend(["--model", args.dl_model])
        print("\n" + "#" * 70)
        print(f"# Running stage {i}: {script.name}")
        print("#" * 70)
        r = subprocess.run(cmd, cwd=str(proj))
        if r.returncode != 0:
            print(f"Stage {i} failed with code {r.returncode}")
            sys.exit(r.returncode)
    print("\nAll requested stages finished OK.")


if __name__ == "__main__":
    main()
