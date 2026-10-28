"""Nested 3–4 path interpretability helpers."""

from scripts.utils.path_interpret import (
    SUPPORT_RATIO_MAX,
    atom_gloss,
    contiguous_subpaths,
    grow_parent,
    is_redundant,
    path_narrative,
    stay_contains_path,
)


def test_contiguous_subpaths_and_parent():
    p3 = ("a|B1|升高", "b|B2|下降", "c|B4|升高")
    assert grow_parent(p3) == ("a|B1|升高", "b|B2|下降")
    subs = contiguous_subpaths(p3)
    assert ("a|B1|升高", "b|B2|下降") in subs
    assert ("b|B2|下降", "c|B4|升高") in subs
    assert p3 not in subs
    p4 = p3 + ("d|低位波动型|B4",)
    sub4 = contiguous_subpaths(p4)
    assert p3 in sub4
    assert len([s for s in sub4 if len(s) == 2]) == 3


def test_redundant_if_almost_same_support_or_tiny_gain():
    assert is_redundant(support_long=90, support_short=100, auprc_gain=0.02, min_gain=0.001)
    assert SUPPORT_RATIO_MAX == 0.90
    assert is_redundant(support_long=50, support_short=100, auprc_gain=0.0005, min_gain=0.001)
    assert not is_redundant(support_long=50, support_short=100, auprc_gain=0.01, min_gain=0.001)


def test_stay_contains_requires_increasing_stages():
    atoms = [(0, "mbp|缓降持平型|B1", False), (2, "lactate|B3|升高", False)]
    assert stay_contains_path(atoms, ("mbp|缓降持平型|B1", "lactate|B3|升高"))
    assert not stay_contains_path(atoms, ("lactate|B3|升高", "mbp|缓降持平型|B1"))


def test_narrative_readable():
    text = path_narrative(("lactate|B1|升高", "urine_output_ml|低位缓升型|B3"), sign=1)
    assert "乳酸" in text and "尿量" in text and "升高" in text
    assert atom_gloss("gcs_total|B4|下降") == "GCS在B4下降"
