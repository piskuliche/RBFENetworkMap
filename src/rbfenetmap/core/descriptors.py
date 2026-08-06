"""Edge descriptors: the single place where scoring inputs are computed.

Every number a scorer sees originates here. That centralisation is deliberate. Scorers
receive a plain ``Mapping[str, float]`` and never import RDKit, which means re-scoring a
network under new weights requires no remapping, a scorer can be tested against
hand-written dictionaries, and a third-party scorer cannot quietly grow a dependency on
how the mapping was produced.

Descriptors are raw, unnormalised, and unweighted. Turning them into a cost -- deciding
that eight soft-core atoms is "one unit of bad" -- is the scorer's job, not this module's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rdkit.Chem import Crippen, rdMolDescriptors

from rbfenetmap.core.kabsch import core_rmsd, pair_distances
from rbfenetmap.core.models import AtomMapping, Ligand, SoftcoreRepair
from rbfenetmap.core.molgraph import ring_systems

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit import Chem

__all__ = ("DESCRIPTOR_NAMES", "compute_descriptors")

#: Every key :func:`compute_descriptors` produces. Scorers use this to validate weights.
DESCRIPTOR_NAMES = (
    "n_core_heavy",
    "n_softcore_heavy_1",
    "n_softcore_heavy_2",
    "n_softcore_max_heavy",
    "softcore_asymmetry",
    "n_heavy_1",
    "n_heavy_2",
    "heavy_atom_delta",
    "charge_delta",
    "n_rings_1",
    "n_rings_2",
    "ring_delta",
    "n_ring_atoms_in_softcore",
    "mcs_fraction",
    "core_rmsd",
    "core_max_pair_distance",
    "rotatable_delta",
    "logp_delta",
    "n_fragments_before_1",
    "n_fragments_before_2",
    "n_demoted_atoms",
)


def _coords(mol: "Chem.Mol", indices: tuple[int, ...]) -> np.ndarray:
    """Return the ``(n, 3)`` coordinates of *indices*."""
    if not indices:
        return np.zeros((0, 3), dtype=float)
    conformer = mol.GetConformer()
    return np.array([list(conformer.GetAtomPosition(int(i))) for i in indices], dtype=float)


def _ring_atoms_in_softcore(mol: "Chem.Mol", softcore: tuple[int, ...]) -> int:
    """Count soft-core atoms belonging to a ring."""
    ring_atoms: set[int] = set()
    for ring in ring_systems(mol):
        ring_atoms |= ring
    return len(ring_atoms & set(softcore))


def compute_descriptors(
    source: Ligand, target: Ligand, mapping: AtomMapping, repair: SoftcoreRepair | None = None
) -> dict[str, float]:
    """Compute every scoring descriptor for one candidate transformation.

    Parameters
    ----------
    source, target : Ligand
        The two ligands.
    mapping : AtomMapping
        The mapping *after* repair, so soft-core sizes reflect what will actually run.
    repair : SoftcoreRepair, optional
        The repair record, contributing the fragmentation and demotion counts.

    Returns
    -------
    dict[str, float]
        Keyed by :data:`DESCRIPTOR_NAMES`. Every value is a float, including the counts,
        so scorers never have to think about integer division.

    Notes
    -----
    Soft-core sizes are counted in **heavy atoms**. Hydrogens follow their parent heavy
    atom into the soft-core automatically, so including them would mostly measure how
    hydrogenated a substituent is rather than how large the perturbation is -- a
    ``-CH3`` to ``-CF3`` change would look like a shrinking soft-core.

    ``core_rmsd`` is measured in place, without superposition. Ligands are normally
    supplied already posed in a common binding-site frame, and in that frame the
    deviation of mapped core atoms says whether the mapping pairs atoms that occupy the
    same part of the pocket. Superposing first would hide exactly that.
    """
    heavy_1 = set(source.heavy_indices)
    heavy_2 = set(target.heavy_indices)

    softcore_heavy_1 = len(heavy_1 & set(mapping.sc1))
    softcore_heavy_2 = len(heavy_2 & set(mapping.sc2))
    core_heavy = len(heavy_1 & set(mapping.cc1))

    coords_1 = _coords(source.mol, mapping.cc1)
    coords_2 = _coords(target.mol, mapping.cc2)
    distances = pair_distances(coords_1, coords_2)

    n_rings_1 = len(ring_systems(source.mol))
    n_rings_2 = len(ring_systems(target.mol))

    smaller_heavy = min(source.n_heavy, target.n_heavy) or 1

    descriptors = {
        "n_core_heavy": float(core_heavy),
        "n_softcore_heavy_1": float(softcore_heavy_1),
        "n_softcore_heavy_2": float(softcore_heavy_2),
        "n_softcore_max_heavy": float(max(softcore_heavy_1, softcore_heavy_2)),
        "softcore_asymmetry": float(abs(softcore_heavy_1 - softcore_heavy_2)),
        "n_heavy_1": float(source.n_heavy),
        "n_heavy_2": float(target.n_heavy),
        "heavy_atom_delta": float(abs(source.n_heavy - target.n_heavy)),
        "charge_delta": float(abs(source.charge - target.charge)),
        "n_rings_1": float(n_rings_1),
        "n_rings_2": float(n_rings_2),
        "ring_delta": float(abs(n_rings_1 - n_rings_2)),
        "n_ring_atoms_in_softcore": float(
            max(_ring_atoms_in_softcore(source.mol, mapping.sc1), _ring_atoms_in_softcore(target.mol, mapping.sc2))
        ),
        "mcs_fraction": float(core_heavy / smaller_heavy),
        "core_rmsd": float(core_rmsd(coords_1, coords_2)),
        "core_max_pair_distance": float(distances.max()) if distances.size else 0.0,
        "rotatable_delta": float(
            abs(rdMolDescriptors.CalcNumRotatableBonds(source.mol) - rdMolDescriptors.CalcNumRotatableBonds(target.mol))
        ),
        "logp_delta": float(abs(Crippen.MolLogP(source.mol) - Crippen.MolLogP(target.mol))),
        "n_fragments_before_1": float(repair.n_fragments_before[0]) if repair else 0.0,
        "n_fragments_before_2": float(repair.n_fragments_before[1]) if repair else 0.0,
        "n_demoted_atoms": float(repair.n_demoted) if repair else 0.0,
    }

    missing = set(DESCRIPTOR_NAMES) - set(descriptors)
    if missing:  # pragma: no cover - guards DESCRIPTOR_NAMES against drifting out of sync
        raise RuntimeError(f"compute_descriptors did not produce declared descriptor(s) {sorted(missing)}.")
    return descriptors
