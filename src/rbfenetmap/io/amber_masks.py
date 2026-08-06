"""Amber ``timask`` / ``scmask`` generation.

A ParmEd-free port of ``BuildEdges._build_rbfe_mask_variables``. It lives in
:mod:`rbfenetmap.io` rather than the core because Amber masks are one output format
among several, and nothing in the planning pipeline should depend on them.

Two correctness traps from the original are preserved deliberately, and both are worth
understanding before touching this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rbfenetmap.core.exceptions import ExporterError
from rbfenetmap.core.models import AtomMapping, Ligand

__all__ = ("DEFAULT_RESIDUE_NAMES", "AmberMasks", "build_amber_masks")

#: Residue names the exporter assigns to the two endpoints of an edge.
DEFAULT_RESIDUE_NAMES = ("SRC", "DST")


@dataclass(frozen=True)
class AmberMasks:
    """The four Amber mask strings for one transformation."""

    timask1: str
    timask2: str
    scmask1: str
    scmask2: str

    def as_dict(self) -> dict[str, str]:
        """Return the masks keyed by their Amber input names."""
        return {"timask1": self.timask1, "timask2": self.timask2, "scmask1": self.scmask1, "scmask2": self.scmask2}


def build_amber_masks(
    source: Ligand, target: Ligand, mapping: AtomMapping, *, residue_names: tuple[str, str] = DEFAULT_RESIDUE_NAMES
) -> AmberMasks:
    """Build the ``timask``/``scmask`` pair for one transformation.

    Parameters
    ----------
    source, target : Ligand
    mapping : AtomMapping
        The repaired mapping.
    residue_names : tuple[str, str], optional
        Residue names for the two endpoints.

    Returns
    -------
    AmberMasks

    Raises
    ------
    rbfenetmap.core.exceptions.ExporterError
        If a soft-core atom name collides with a common-core atom name, or the mapping
        violates Amber's linear-scaling constraint.

    Notes
    -----
    **Trap one: name collisions.** Amber soft-core masks select atoms *by name*, not by
    index. If a soft-core atom shares its name with a common-core atom in the same
    residue, the mask silently selects that core atom too, and the run proceeds with a
    soft-core region larger than intended. Nothing downstream reports this; the free
    energies are simply wrong. Atom names must therefore be unique within a ligand --
    which is what ``antechamber -du y`` produces.

    **Trap two: linear scaling.** Amber requires
    ``len(TI1) - len(SC1) == len(TI2) - len(SC2)``. Since each side of that reduces to
    the size of the common core, the constraint is exactly ``len(cc1) == len(cc2)``
    together with injectivity of ``cc2`` -- and both are already invariants of
    :class:`~rbfenetmap.core.models.AtomMapping`. A mapping built by this package cannot
    violate it, so the check below can only ever fire on a hand-authored map. It is kept
    because it is cheap, and because failing here with a clear message beats failing
    inside ``pmemd`` with an opaque one.
    """
    residue_1, residue_2 = f":{residue_names[0]}", f":{residue_names[1]}"
    names_1 = source.atom_names
    names_2 = target.atom_names

    softcore_names_1 = sorted({names_1[i] for i in mapping.sc1})
    softcore_names_2 = sorted({names_2[i] for i in mapping.sc2})

    for side, ligand, names, softcore_names, core_indices in (
        ("source", source, names_1, softcore_names_1, mapping.cc1),
        ("target", target, names_2, softcore_names_2, mapping.cc2),
    ):
        core_names = {names[i] for i in core_indices}
        collisions = sorted(set(softcore_names) & core_names)
        if collisions:
            raise ExporterError(
                f"{ligand.name} ({side}): soft-core atom name(s) {collisions} are also used by "
                "common-core atoms. Amber soft-core masks select by name, so those core atoms would "
                "be pulled into the soft-core region without any warning at run time. Ensure atom "
                "names are unique within the ligand (antechamber's `-du y` does this)."
            )

    linear_1 = mapping.n_atoms_1 - len(mapping.sc1)
    linear_2 = mapping.n_atoms_2 - len(mapping.sc2)
    if linear_1 != linear_2:  # pragma: no cover - unreachable for mappings built by this package
        duplicates = sorted(i for i, n in Counter(mapping.cc2).items() if n > 1)
        detail = f" The atom map is not injective (duplicate target indices: {duplicates})." if duplicates else ""
        raise ExporterError(
            "Amber linear-scaling constraint violated: "
            f"len(TI1)-len(SC1) = {mapping.n_atoms_1}-{len(mapping.sc1)} = {linear_1} but "
            f"len(TI2)-len(SC2) = {mapping.n_atoms_2}-{len(mapping.sc2)} = {linear_2}.{detail}"
        )

    return AmberMasks(
        timask1=residue_1,
        timask2=residue_2,
        scmask1=f"{residue_1}&@{','.join(softcore_names_1)}" if softcore_names_1 else "",
        scmask2=f"{residue_2}&@{','.join(softcore_names_2)}" if softcore_names_2 else "",
    )
