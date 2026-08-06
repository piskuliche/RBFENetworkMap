"""Geometry-based atom mapping.

A port of ``amberstudio.worknodes.cartograph``, with the two unused ``parmed.Structure``
positionals dropped from the entry point -- which is what keeps ParmEd out of this
package's dependency list.

The approach is geometric rather than topological: shape-align the two molecules, pair
atoms by proximity via the Hungarian algorithm, then apply a chain of topology filters
that reject pairings no alchemical transformation should make. Where an MCS asks "what
substructure do these molecules share", this asks "which atoms occupy the same space",
which is usually the better question for ligands already posed in a binding site.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from scipy.optimize import linear_sum_assignment

from rbfenetmap.core.coreprune import prune_core
from rbfenetmap.core.exceptions import MappingError
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.models import AtomMapping, Ligand
from rbfenetmap.core.molgraph import connected_components_of, mol_to_graph
from rbfenetmap.core.options import MappingOptions

__all__ = ("COMMON_CORE_DISTANCE_THRESHOLD_ANGSTROM", "CartographMapper", "cartograph_mapping")

#: Default geometric cutoff, in angstroms, for candidate atom pairings.
COMMON_CORE_DISTANCE_THRESHOLD_ANGSTROM = 2.0


def _align_mol_shape(mol: Chem.Mol, reference: Chem.Mol) -> Chem.Mol:
    """Return a copy of *mol* O3A-shape-aligned onto *reference*."""
    aligned = Chem.Mol(mol)
    rdMolAlign.GetO3A(prbMol=aligned, refMol=reference).Align()
    return aligned


def _coords(mol: Chem.Mol) -> np.ndarray:
    """Return the ``(n, 3)`` conformer coordinates of *mol*."""
    conformer = mol.GetConformer()
    return np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)


def _filter_hydrogens(mol_1: Chem.Mol, mol_2: Chem.Mol, mapping: dict[int, int]) -> dict[int, int]:
    """Keep only hydrogen-to-hydrogen and heavy-to-heavy pairings."""
    return {
        i: j
        for i, j in mapping.items()
        if (mol_1.GetAtomWithIdx(i).GetAtomicNum() == 1) == (mol_2.GetAtomWithIdx(j).GetAtomicNum() == 1)
    }


def _filter_ring_breaks(mol_1: Chem.Mol, mol_2: Chem.Mol, mapping: dict[int, int]) -> dict[int, int]:
    """Reject pairings that move an atom between ring and non-ring."""
    return {
        i: j for i, j in mapping.items() if mol_1.GetAtomWithIdx(i).IsInRing() == mol_2.GetAtomWithIdx(j).IsInRing()
    }


def _filter_ring_sizes(mol_1: Chem.Mol, mol_2: Chem.Mol, mapping: dict[int, int]) -> dict[int, int]:
    """Reject ring-atom pairings that share no ring size."""
    rings_1 = mol_1.GetRingInfo()
    rings_2 = mol_2.GetRingInfo()
    kept: dict[int, int] = {}
    for i, j in mapping.items():
        atom_1 = mol_1.GetAtomWithIdx(i)
        atom_2 = mol_2.GetAtomWithIdx(j)
        if not (atom_1.IsInRing() and atom_2.IsInRing()) or (
            set(rings_1.AtomRingSizes(i)) & set(rings_2.AtomRingSizes(j))
        ):
            kept[i] = j
    return kept


def _filter_whole_rings(mol_1: Chem.Mol, mol_2: Chem.Mol, mapping: dict[int, int]) -> dict[int, int]:
    """Keep ring atoms only when their entire ring is mapped.

    Iterated to a fixpoint: dropping one ring atom can leave a neighbouring fused ring
    incompletely mapped, which must then be dropped in turn.
    """
    proposed = dict(mapping)
    changed = True
    while changed:
        changed = False
        for mol, side in ((mol_1, 1), (mol_2, 2)):
            mapped = set(proposed) if side == 1 else set(proposed.values())
            drop: set[int] = set()
            for ring in mol.GetRingInfo().AtomRings():
                ring_set = set(ring)
                if ring_set & mapped and not ring_set <= mapped:
                    drop |= ring_set & mapped
            if drop:
                if side == 1:
                    proposed = {i: j for i, j in proposed.items() if i not in drop}
                else:
                    proposed = {i: j for i, j in proposed.items() if j not in drop}
                changed = True
    return proposed


def _filter_bond_breaks(mol_1: Chem.Mol, mol_2: Chem.Mol, mapping: dict[int, int]) -> dict[int, int]:
    """Reject pairings that would require breaking a bond within the common core.

    Two mapped atoms bonded on one side must be bonded on the other. A core that
    silently changes its own connectivity is not a common core.
    """
    proposed = dict(mapping)
    changed = True
    while changed:
        changed = False
        for i, j in list(proposed.items()):
            for neighbour in mol_1.GetAtomWithIdx(i).GetNeighbors():
                partner = proposed.get(neighbour.GetIdx())
                if partner is None:
                    continue
                if mol_2.GetBondBetweenAtoms(j, partner) is None:
                    proposed.pop(i, None)
                    changed = True
                    break
    return proposed


def _largest_connected_core(mol_1: Chem.Mol, mol_2: Chem.Mol, mapping: dict[int, int]) -> dict[int, int]:
    """Keep only the largest connected component of the mapped core.

    A common core made of several disconnected islands cannot be held rigid as a unit.
    """
    if not mapping:
        return mapping
    components = connected_components_of(mol_to_graph(mol_1), set(mapping))
    if len(components) <= 1:
        return mapping
    keep = components[0]
    return {i: j for i, j in mapping.items() if i in keep}


#: Applied in order; each filter sees the output of the previous one.
_FILTERS = (
    _filter_hydrogens,
    _filter_ring_breaks,
    _filter_ring_sizes,
    _filter_whole_rings,
    _filter_bond_breaks,
    _largest_connected_core,
)


def cartograph_mapping(
    mol_1: Chem.Mol, mol_2: Chem.Mol, *, distance_threshold: float = COMMON_CORE_DISTANCE_THRESHOLD_ANGSTROM
) -> dict[int, int]:
    """Return the geometric atom correspondence between two molecules.

    Parameters
    ----------
    mol_1, mol_2 : rdkit.Chem.Mol
        Molecules with 3D conformers.
    distance_threshold : float, optional
        Candidate pairings further apart than this, after shape alignment, are discarded
        before the topology filters run.

    Returns
    -------
    dict[int, int]
        ``{index_in_mol_1: index_in_mol_2}``.
    """
    aligned = _align_mol_shape(mol_2, mol_1)
    coords_1 = _coords(mol_1)
    coords_2 = _coords(aligned)

    distances = np.linalg.norm(coords_1[:, None, :] - coords_2[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(distances)
    mapping = {int(r): int(c) for r, c in zip(rows, cols) if distances[r, c] <= distance_threshold}

    for filter_fn in _FILTERS:
        mapping = filter_fn(mol_1, mol_2, mapping)
    return mapping


class CartographMapper(AbstractMapper):
    """Geometry-based mapper: shape alignment plus Hungarian assignment."""

    name: ClassVar[str] = "cartograph"

    def map_pair(self, source: Ligand, target: Ligand, options: MappingOptions) -> AtomMapping:
        """Return the geometric correspondence between *source* and *target*.

        Raises
        ------
        rbfenetmap.core.exceptions.MappingError
            If shape alignment fails, or nothing survives the topology filters.
        """
        try:
            core = cartograph_mapping(source.mol, target.mol, distance_threshold=options.distance_threshold)
        except (RuntimeError, ValueError) as exc:
            raise MappingError(f"{source.name}~{target.name}: geometric mapping failed: {exc}") from exc
        if not core:
            raise MappingError(
                f"{source.name}~{target.name}: no atom pairs survived the geometric cutoff of "
                f"{options.distance_threshold} A and the topology filters. The ligands are probably "
                "not posed in a common frame."
            )
        mapping = AtomMapping.from_core_pairs(
            core, n_atoms_1=source.n_atoms, n_atoms_2=target.n_atoms, method=self.name
        )
        return prune_core(source, target, mapping, options.core_pruning)
