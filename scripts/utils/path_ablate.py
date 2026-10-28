"""Pre-specified path-set ablation arms (docs/20). No val/holdout selection."""

from __future__ import annotations

from scripts.utils.path_protocol import parse_path_key


def is_lab_atom(atom: str) -> bool:
    a = str(atom)
    return a.endswith("|升高") or a.endswith("|下降")


def path_has_lab(atoms: tuple[str, ...]) -> bool:
    return any(is_lab_atom(a) for a in atoms)


def path_is_motif_only(atoms: tuple[str, ...]) -> bool:
    return bool(atoms) and not path_has_lab(atoms)


def prefix_paths(keys: list[str], k: int) -> list[tuple[str, ...]]:
    return [parse_path_key(p) for p in keys[: int(k)]]


def filter_by_sign(keys: list[str], sign_of: dict[str, int], want: int) -> list[tuple[str, ...]]:
    out = []
    for k in keys:
        if int(sign_of.get(k, 0)) == int(want):
            out.append(parse_path_key(k))
    return out


def filter_motif_only(keys: list[str]) -> list[tuple[str, ...]]:
    return [parse_path_key(k) for k in keys if path_is_motif_only(parse_path_key(k))]


def filter_has_lab(keys: list[str]) -> list[tuple[str, ...]]:
    return [parse_path_key(k) for k in keys if path_has_lab(parse_path_key(k))]


def leave_one_out(keys: list[str], drop: int) -> list[tuple[str, ...]]:
    return [parse_path_key(k) for i, k in enumerate(keys) if i != int(drop)]
