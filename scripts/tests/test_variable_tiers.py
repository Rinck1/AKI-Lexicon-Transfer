"""Tier registry sanity."""

from scripts.utils.variable_tiers import list_tier, protocol_tier_map


def test_tiers_disjoint_a_b():
    a = set(list_tier("A"))
    b = set(list_tier("B"))
    assert a.isdisjoint(b)


def test_core_shapelet_vars_present():
    a = set(list_tier("A"))
    for v in ["creatinine", "bun", "urine_output_ml", "sbp", "gcs_total"]:
        assert v in a


def test_protocol_map_covers_a():
    m = protocol_tier_map()
    assert m["creatinine"] == "A"
    assert m["albumin"] == "B"
