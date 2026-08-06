"""Pre-repair demotion of mapped atom pairs that should not be common core.

An MCS is a topological answer to a topological question, and it will happily map a
carbon onto a nitrogen or a ``CH2`` onto a ``CH3`` because the graphs match. Whether such
a pair may be held fixed through an alchemical transformation is a different question,
and the answer depends on what the user is willing to accept.

This module generalizes ``BuildEdges._classify_softcore_method0/1/2`` from a three-way
method string into independent flags on
:class:`~rbfenetmap.core.options.CorePruningPolicy`, with presets reproducing the
original three. It runs before :mod:`rbfenetmap.core.softcore`, so anything demoted here
becomes part of the fragmentation problem the repair then has to solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from rbfenetmap.core.models import AtomMapping, Ligand
from rbfenetmap.core.molgraph import acyclic_branches, mol_to_graph
from rbfenetmap.core.options import CorePruningPolicy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit import Chem

__all__ = ("choose_softcore_branch", "prune_core")


def choose_softcore_branch(graph: nx.Graph, core: set[int], center: int) -> set[int]:
    """Return the atoms of the least-conserved branches hanging off *center*.

    Partitions the graph into the branches reachable from *center* through **acyclic**
    bonds, counts how many common-core atoms each branch holds, and returns the atoms of
    every branch below the maximum.

    Parameters
    ----------
    graph : networkx.Graph
        Bond graph of the molecule.
    core : set[int]
        Atom indices currently in the common core.
    center : int
        The atom whose substituents are being classified.

    Returns
    -------
    set[int]
        Atoms to demote to the soft-core.

    Notes
    -----
    This is a corrected port of ``BuildEdges._choose_sc_atoms``. The original partitions
    across *every* bond partner via ``_atoms_beyond_bond``, which blocks only the central
    atom. For a ring atom that traversal wraps around the ring and returns, so each
    "branch" contains nearly the whole molecule and the branches overlap almost
    completely. Demoting all-but-the-largest then demotes almost everything.

    Restricting the partition to acyclic bonds keeps the branches genuinely disjoint. A
    ring atom simply yields fewer branches -- possibly none, in which case nothing is
    demoted here and the whole-ring closure rule in :mod:`rbfenetmap.core.softcore` deals
    with the ring.
    """
    branches = acyclic_branches(graph, center)
    if len(branches) < 2:
        return set()
    counts = {neighbour: len(atoms & core) for neighbour, atoms in branches.items()}
    best = max(counts.values())
    demoted: set[int] = set()
    for neighbour, atoms in branches.items():
        if counts[neighbour] < best:
            demoted |= atoms
    return demoted


def _pair_mismatches(atom_1: "Chem.Atom", atom_2: "Chem.Atom", policy: CorePruningPolicy) -> bool:
    """Whether a mapped pair violates any enabled same-property rule."""
    if policy.demote_element_mismatch and atom_1.GetAtomicNum() != atom_2.GetAtomicNum():
        return True
    if policy.demote_degree_mismatch and atom_1.GetDegree() != atom_2.GetDegree():
        return True
    if policy.demote_formal_charge_mismatch and atom_1.GetFormalCharge() != atom_2.GetFormalCharge():
        return True
    if policy.demote_aromaticity_mismatch and atom_1.GetIsAromatic() != atom_2.GetIsAromatic():
        return True
    if policy.demote_ring_membership_mismatch and atom_1.IsInRing() != atom_2.IsInRing():
        return True
    return False


def prune_core(
    source: Ligand, target: Ligand, mapping: AtomMapping, policy: CorePruningPolicy | None = None
) -> AtomMapping:
    """Demote mapped pairs that the policy says cannot be held in common.

    Parameters
    ----------
    source, target : Ligand
        The two ligands.
    mapping : AtomMapping
        The mapper's raw correspondence.
    policy : CorePruningPolicy, optional
        Which demotion rules to apply. Defaults are used if omitted.

    Returns
    -------
    AtomMapping
        A mapping with the offending pairs moved into the soft-core. Returned unchanged
        when nothing is demoted, so the common case costs almost nothing.

    Notes
    -----
    A light-element swap (a hydrogen mapped onto a heavy atom) is handled specially:
    demoting just the pair would leave the heavy atom's substituents attached to a
    soft-core atom while remaining common core themselves. The branch-selection helper
    takes the less-conserved substituents along, which is what
    ``_classify_softcore_method1/2`` intends.
    """
    policy = policy or CorePruningPolicy()
    forward = mapping.forward
    if not forward:
        return mapping

    graph_1 = mol_to_graph(source.mol)
    graph_2 = mol_to_graph(target.mol)
    core_1 = set(mapping.cc1)
    core_2 = set(mapping.cc2)

    demote_1: set[int] = set()
    demote_2: set[int] = set()

    for index_1, index_2 in forward.items():
        atom_1 = source.mol.GetAtomWithIdx(index_1)
        atom_2 = target.mol.GetAtomWithIdx(index_2)
        z1, z2 = atom_1.GetAtomicNum(), atom_2.GetAtomicNum()

        if policy.demote_light_element_swap and (z1 == 1) != (z2 == 1):
            # A hydrogen paired with a heavy atom. Take the less-conserved substituents
            # of the heavy side with it, so the boundary lands somewhere chemically
            # sensible rather than mid-substituent.
            demote_1.add(index_1)
            demote_2.add(index_2)
            if z1 != 1:
                demote_1 |= choose_softcore_branch(graph_1, core_1, index_1)
            if z2 != 1:
                demote_2 |= choose_softcore_branch(graph_2, core_2, index_2)
            continue

        if _pair_mismatches(atom_1, atom_2, policy):
            demote_1.add(index_1)
            demote_2.add(index_2)

    if not demote_1 and not demote_2:
        return mapping

    return AtomMapping.from_core_pairs(
        {a: b for a, b in forward.items() if a not in demote_1 and b not in demote_2},
        n_atoms_1=mapping.n_atoms_1,
        n_atoms_2=mapping.n_atoms_2,
        method=mapping.method,
    )
