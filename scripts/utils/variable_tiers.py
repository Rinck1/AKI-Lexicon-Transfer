"""Three-tier variable pools (protocol §3)."""

from __future__ import annotations

from typing import Dict, List

from scripts.utils.constants import TIER_A_SHAPELET, TIER_B_SPARSE, TIER_C_STATIC


def protocol_tier_map() -> Dict[str, str]:
    """Map variable name → protocol clinical hint A/B/C."""
    m: Dict[str, str] = {}
    for v in TIER_A_SHAPELET:
        m[v] = "A"
    for v in TIER_B_SPARSE:
        m[v] = "B"
    for v in TIER_C_STATIC:
        m[v] = "C"
    return m


def list_tier(tier: str) -> List[str]:
    tier = tier.upper()
    if tier == "A":
        return list(TIER_A_SHAPELET)
    if tier == "B":
        return list(TIER_B_SPARSE)
    if tier == "C":
        return list(TIER_C_STATIC)
    raise ValueError(f"Unknown tier: {tier}")


# Clinical variable groups for selection (atomic units)
VARIABLE_GROUPS: Dict[str, List[str]] = {
    "creatinine": ["creatinine", "creatinine_delta_6h", "creatinine_slope_6h"],
    "bun": ["bun", "bun_delta_6h", "bun_slope_6h"],
    "urine": ["urine_output_ml", "urine_oliguria_flag"],
    "sbp": ["sbp", "sbp_delta_6h", "sbp_roll_mean_6h"],
    "heart_rate": ["heart_rate", "heart_rate_delta_6h"],
    "spo2": ["spo2", "spo2_delta_6h"],
    "gcs": ["gcs_total", "gcs_motor", "gcs_eye", "gcs_verbal"],
    "lactate": ["lactate"],
    "bicarbonate": ["bicarbonate"],
}
