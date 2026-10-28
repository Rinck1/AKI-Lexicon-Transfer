#!/usr/bin/env python3
"""Run remaining path-module closeout in order. Safe to re-run: 26 refuses a second holdout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(script: str) -> None:
    cmd = [PY, "-u", str(PROJ / "scripts" / script)]
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(PROJ))


def main() -> None:
    hold = PROJ / "data/processed/paths/path_final_7var_motif84_labs11/path_holdout_reproduce.json"
    if not hold.is_file():
        print("[closeout] holdout not done — running 26 (one shot)", flush=True)
        run("26_reproduce_paths_holdout.py")
    else:
        print("[closeout] holdout already exists, skip 26", flush=True)
    run("27_audit_selected_paths.py")
    run("29_ablate_path_set.py")
    run("30_summarize_gnn_ablations.py")
    print("[closeout] next (manual, needs new data): python -u scripts/28_external_paths_stub.py --cohort-dir data/external/eicu")


if __name__ == "__main__":
    main()
