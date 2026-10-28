"""
AKI Research Project — Pytest Configuration & Fixtures
=======================================================
"""

import os
import sys
import pytest

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def mimic_dir():
    """Return the MIMIC-III data directory."""
    from scripts.utils.constants import MIMIC_DIR
    assert os.path.isdir(MIMIC_DIR), f"MIMIC-III directory not found: {MIMIC_DIR}"
    return MIMIC_DIR


@pytest.fixture(scope="session")
def aki_icd9_codes():
    """Return the AKI ICD9 code list."""
    from scripts.utils.constants import AKI_ICD9_CODES
    return list(AKI_ICD9_CODES.keys())
