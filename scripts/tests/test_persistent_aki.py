from scripts.utils.persistent_aki import creatinine_resolving


def test_resolving_absolute_drop():
    assert creatinine_resolving(2.0, 1.6) == 1.0


def test_nonresolving_small_drop():
    assert creatinine_resolving(2.0, 1.9) == 0.0


def test_resolving_relative_25pct():
    # 4.0 -> 3.0 is 25% drop, even if absolute drop is 1.0
    assert creatinine_resolving(4.0, 3.0) == 1.0


def test_missing_returns_nan():
    import math

    assert math.isnan(creatinine_resolving(float("nan"), 1.0))
    assert math.isnan(creatinine_resolving(1.0, float("nan")))
