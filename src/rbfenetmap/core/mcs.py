"""The maximum-common-substructure search, in one place.

``FindMCS`` has a dozen switches and the answer changes with every one of them. Two callers
configuring it separately will drift, and the drift is invisible: an aligner that maximises
one substructure while the geometry gate measures a different one produces a run whose
alignment report looks healthy next to edges rejected for ``core_geometry_mismatch``, with
nothing on screen to reconcile the two.

So the settings live here and both :class:`~rbfenetmap.plugins.mappers.mcss_mapper.MCSSMapper`
and :mod:`rbfenetmap.core.align` call in. This module holds no policy of its own -- every
switch comes from the caller's :class:`~rbfenetmap.core.options.MappingOptions`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Chem import rdFMCS

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from rbfenetmap.core.options import MappingOptions

__all__ = ("mcs_embeddings", "mcs_query")


def mcs_query(
    mol_1: "Chem.Mol", mol_2: "Chem.Mol", options: "MappingOptions", *, match_elements: bool = False
) -> "Chem.Mol | None":
    """Return the MCS of two molecules as a query molecule.

    Parameters
    ----------
    mol_1, mol_2 : rdkit.Chem.Mol
    options : MappingOptions
        Supplies ``timeout``, the four ``FindMCS`` comparison flags, and the ring settings.
    match_elements : bool, optional
        Require paired atoms to be the same element. Default ``False``, which is what a
        mapper wants. See the note below before switching it on or off.

    Returns
    -------
    rdkit.Chem.Mol or None
        ``None`` when the molecules share no substructure, or when the SMARTS RDKit
        produces cannot be parsed back into a query. Both are ordinary outcomes for a
        caller deciding what to do about a pair, so neither raises here.

    Notes
    -----
    ``bondCompare=CompareAny`` is not laxness, and is not negotiable per caller. Ligands
    routinely arrive from force-field topologies whose bond orders are approximate -- an
    Amber mol2 recording a carbonyl as ``C-O`` single is the everyday case, and
    :func:`rbfenetmap.io.loaders._prepare` deliberately preserves it rather than
    "correcting" it. A strict bond compare would silently shrink the common substructure on
    exactly those inputs.

    The *atom* comparison is a different matter, and is the one setting callers legitimately
    disagree about. A mapper can afford ``CompareAny`` because everything downstream of it
    is a safety net: :func:`~rbfenetmap.core.coreprune.prune_core` demotes element
    mismatches out of the core, and the geometry gate catches whatever survives. A caller
    with no such net -- alignment, where the correspondence *is* the answer and nothing
    revisits it -- cannot. Left permissive there, ``FindMCS`` will pair a methoxy oxygen
    with a methyl carbon and an amide nitrogen with a hydrogen, which superposes eleven
    atoms to a convincing fraction of an angstrom while putting the scaffold several
    angstroms wrong. It is precisely the failure a good-looking RMSD hides.
    """
    result = rdFMCS.FindMCS(
        [mol_1, mol_2],
        maximizeBonds=False,
        threshold=1.0,
        matchValences=options.match_valences,
        ringMatchesRingOnly=options.ring_matches_ring_only,
        completeRingsOnly=options.complete_rings_only,
        matchChiralTag=options.match_chiral_tag,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareElements if match_elements else rdFMCS.AtomCompare.CompareAny,
        timeout=options.timeout,
    )
    if not result.smartsString:
        return None
    return Chem.MolFromSmarts(result.smartsString)


def mcs_embeddings(
    mol_1: "Chem.Mol", mol_2: "Chem.Mol", pattern: "Chem.Mol", options: "MappingOptions"
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return every embedding of *pattern* in each molecule.

    Parameters
    ----------
    mol_1, mol_2 : rdkit.Chem.Mol
    pattern : rdkit.Chem.Mol
        A query molecule, normally from :func:`mcs_query`.
    options : MappingOptions
        Supplies ``max_matches``.

    Returns
    -------
    tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]
        The embeddings in ``mol_1`` and in ``mol_2``. Either can be empty.

    Notes
    -----
    ``uniquify=False`` is what makes the symmetry visible. With it set, RDKit collapses
    embeddings related by an automorphism of the query and returns one representative -- so
    a *para*-substituted ring offers a single embedding and the flip that pairs atoms across
    the ring from one another can never be examined, let alone rejected.
    """
    return (
        mol_1.GetSubstructMatches(pattern, uniquify=False, maxMatches=options.max_matches),
        mol_2.GetSubstructMatches(pattern, uniquify=False, maxMatches=options.max_matches),
    )
