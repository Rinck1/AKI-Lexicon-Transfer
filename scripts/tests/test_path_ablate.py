"""Path-set ablation arm membership."""

from scripts.utils.path_ablate import (
    filter_has_lab,
    filter_motif_only,
    is_lab_atom,
    leave_one_out,
    path_is_motif_only,
    prefix_paths,
)


def test_lab_vs_motif():
    assert is_lab_atom("lactate|B1|升高")
    assert not is_lab_atom("mbp|缓降持平型|B3")
    assert path_is_motif_only(("urine_output_ml|低位波动型|B1", "urine_output_ml|进行性下降型|B2"))
    keys = [
        "lactate|B1|升高 → urine_output_ml|低位缓升型|B3",
        "urine_output_ml|低位波动型|B1 → urine_output_ml|进行性下降型|B2",
    ]
    assert len(filter_motif_only(keys)) == 1
    assert len(filter_has_lab(keys)) == 1


def test_prefix_and_loo():
    keys = ["a → b", "c → d", "e → f"]
    assert len(prefix_paths(keys, 2)) == 2
    loo = leave_one_out(keys, 0)
    assert loo == [("c", "d"), ("e", "f")]
