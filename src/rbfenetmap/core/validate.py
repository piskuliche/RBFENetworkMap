"""Validation of the common-core / soft-core mapping contract.

A parmed-free port of ``BuildEdges._validate_mapping_result``. Kept in its own module
rather than inlined into :meth:`~rbfenetmap.core.models.AtomMapping.__post_init__` so it
can be called directly on a mapper's raw output during debugging, and so the error
messages live in one place where they can be kept specific.

Every message names the offending indices. A mapping is rejected at construction, so a
vague message here becomes a vague failure hundreds of lines from the actual mistake.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rbfenetmap.core.models import AtomMapping

__all__ = ("validate_mapping",)


def _duplicates(values: Sequence[int]) -> list[int]:
    """Return the values appearing more than once, sorted."""
    return sorted(v for v, n in Counter(values).items() if n > 1)


def _check_index_set(label: str, values: Sequence[int], n_atoms: int, *, note: str = "") -> None:
    """Check *values* are unique, integral, and within ``range(n_atoms)``."""
    dupes = _duplicates(values)
    if dupes:
        raise ValueError(f"{label} contains duplicate atom indices {dupes}.{note}")
    out_of_range = sorted(v for v in values if not 0 <= v < n_atoms)
    if out_of_range:
        raise ValueError(f"{label} contains atom indices {out_of_range} outside range(0, {n_atoms}).")


# Duplicates in cc2 mean several side-1 atoms map onto one side-2 atom. That is the same
# condition as a non-injective correspondence, so the explanation rides along with the
# duplicate check rather than getting its own (unreachable) test further down.
_CC2_INJECTIVITY_NOTE = (
    " Several side-1 atoms map onto the same side-2 atom, so the common-core "
    "correspondence is not injective. This also violates Amber's linear-scaling "
    "constraint len(TI1)-len(SC1) == len(TI2)-len(SC2)."
)


def validate_mapping(mapping: "AtomMapping") -> None:
    """Enforce every invariant of the mapping contract.

    Parameters
    ----------
    mapping : AtomMapping
        The mapping to check.

    Raises
    ------
    ValueError
        With a message naming the specific offending atom indices.

    Notes
    -----
    The partition check (``sc_k`` and ``cc_k`` disjoint and jointly covering every atom)
    is the one that catches the largest class of real mapper bugs. An atom that is in
    neither set has no defined behaviour under the transformation: it is neither held
    fixed nor alchemically transformed. Mappers that forget hydrogens, or that build the
    soft-core from a stale copy of the core, fail exactly here.
    """
    for label, values, n_atoms, note in (
        ("cc1", mapping.cc1, mapping.n_atoms_1, ""),
        ("cc2", mapping.cc2, mapping.n_atoms_2, _CC2_INJECTIVITY_NOTE),
        ("sc1", mapping.sc1, mapping.n_atoms_1, ""),
        ("sc2", mapping.sc2, mapping.n_atoms_2, ""),
    ):
        _check_index_set(label, values, n_atoms, note=note)

    if len(mapping.cc1) != len(mapping.cc2):
        raise ValueError(
            f"Common core sizes disagree: len(cc1)={len(mapping.cc1)} but len(cc2)={len(mapping.cc2)}. "
            "cc1[i] and cc2[i] are corresponding atoms, so the two must be the same length."
        )

    for side, cc, sc, n_atoms in (
        (1, mapping.cc1, mapping.sc1, mapping.n_atoms_1),
        (2, mapping.cc2, mapping.sc2, mapping.n_atoms_2),
    ):
        overlap = sorted(set(cc) & set(sc))
        if overlap:
            raise ValueError(
                f"Side {side}: atoms {overlap} are in both the common core and the soft-core. "
                "An atom is either held in common or transformed, never both."
            )
        missing = sorted(set(range(n_atoms)) - set(cc) - set(sc))
        if missing:
            raise ValueError(
                f"Side {side}: atoms {missing[:16]}{'...' if len(missing) > 16 else ''} are in neither the "
                f"common core nor the soft-core, so the partition is incomplete "
                f"({len(cc)} + {len(sc)} != {n_atoms}). Every atom must have a defined role."
            )

    # Injectivity of cc2 is already guaranteed: the duplicate check above rejects exactly
    # that condition, carrying _CC2_INJECTIVITY_NOTE to explain the consequence. Together
    # with len(cc1) == len(cc2) it means Amber's linear-scaling constraint holds for every
    # mapping this package builds, so the exporter's own check can only fire on a
    # hand-authored map.
