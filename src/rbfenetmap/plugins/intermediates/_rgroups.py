"""Shared R-group decomposition for the built-in intermediate generators.

Both built-in generators answer the same question first: *where do these two ligands
actually differ, and what hangs off the shared core at each of those places?* The answer
is a list of :class:`Position` records, and everything either generator invents is some
choice of which parent's substituent to put at each one.

Factored out of :mod:`~rbfenetmap.plugins.intermediates.fragment_swap` when the PairMap
generator landed rather than copied into it. The decomposition decides which atoms count
as "the same position" on the two parents, and two copies that drifted would mean the two
generators disagreeing about what a molecule *is* while both looking correct in isolation.

Why the correspondence is chosen by geometry
--------------------------------------------

:func:`shared_core` runs the MCS with ``match_elements=True`` and then picks the embedding
with the lowest **in-place** RMSD, exactly as :mod:`rbfenetmap.core.posing` does. A
symmetric ring has several equally good matches on paper, and the wrong one transplants a
substituent to the far side of the ring while producing a perfectly valid molecule. The
coordinates are the only thing that breaks that tie correctly, and the parents are already
co-posed, so they are available.

What counts as a substituent
----------------------------

A branch attached to the core by exactly one bond. A fused ring or a macrocyclic linker
touches the core twice and is skipped: "replace the substituent here" is not a description
of what would have to happen to it, and a generator that pretended otherwise would emit a
molecule no reviewer could relate to either parent. A core atom carrying *two* heavy
branches is skipped for the same reason -- which of the two is "the" substituent is
genuinely ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np
from rdkit import Chem

from rbfenetmap.core.kabsch import core_rmsd
from rbfenetmap.core.mcs import mcs_embeddings, mcs_query
from rbfenetmap.core.models import Ligand
from rbfenetmap.core.options import MappingOptions

__all__ = (
    "HYDROGEN_KEY",
    "Position",
    "Substituent",
    "assemble",
    "branches",
    "differing_positions",
    "first_hydrogen",
    "shared_core",
    "substituents",
)

#: Key used for a position carrying nothing but hydrogen, so an ``H -> CH3`` position
#: compares as a difference like any other.
HYDROGEN_KEY = "[H]"


@dataclass(frozen=True)
class Substituent:
    """One simple branch hanging off a core atom.

    Parameters
    ----------
    attachment : int
        Index of the branch atom bonded to the core.
    atoms : frozenset[int]
        Every atom of the branch, hydrogens included.
    heavy : int
        Heavy atoms in the branch. This is what a link score is measured in, so it is
        computed once here rather than recounted at every scoring call.
    key : str
        Canonical SMILES of the fragment, used only to decide whether two positions carry
        the *same* group.
    """

    attachment: int
    atoms: frozenset[int]
    heavy: int
    key: str


@dataclass(frozen=True)
class Position:
    """One place the two parents differ, and what each of them puts there.

    Parameters
    ----------
    source_anchor, target_anchor : int
        The corresponding core atoms on the two parents.
    source, target : Substituent or None
        The group each parent carries. ``None`` means the position carries nothing but
        hydrogen, which is a group like any other as far as a swap is concerned.

    Notes
    -----
    ``source`` and ``target`` are never *both* ``None`` and never carry the same
    :attr:`Substituent.key`: a position that is not a difference is not a position.
    """

    source_anchor: int
    target_anchor: int
    source: Substituent | None
    target: Substituent | None

    @property
    def source_heavy(self) -> int:
        """Heavy atoms the source parent puts here."""
        return 0 if self.source is None else self.source.heavy

    @property
    def target_heavy(self) -> int:
        """Heavy atoms the target parent puts here."""
        return 0 if self.target is None else self.target.heavy

    @property
    def truncatable(self) -> bool:
        """Whether removing the group here is a third, distinct option.

        ``False`` when one parent already carries only hydrogen there -- truncating would
        reproduce that parent's group, not a new one.
        """
        return self.source is not None and self.target is not None


def branches(mol: Chem.Mol, core: frozenset[int]) -> Iterator[tuple[int, int, frozenset[int]]]:
    """Yield ``(core atom, attachment atom, component)`` for each simple substituent.

    A component attached to the core at more than one bond -- a fused ring, a macrocyclic
    linker -- is skipped rather than reported. Swapping one is not a substituent swap.
    """
    non_core = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in core]
    seen: set[int] = set()
    for start in non_core:
        if start in seen:
            continue
        component: set[int] = set()
        stack = [start]
        while stack:
            idx = stack.pop()
            if idx in component:
                continue
            component.add(idx)
            for neighbour in mol.GetAtomWithIdx(idx).GetNeighbors():
                if neighbour.GetIdx() not in core and neighbour.GetIdx() not in component:
                    stack.append(neighbour.GetIdx())
        seen |= component
        bonds = {
            (neighbour.GetIdx(), idx)
            for idx in component
            for neighbour in mol.GetAtomWithIdx(idx).GetNeighbors()
            if neighbour.GetIdx() in core
        }
        if len(bonds) == 1:
            anchor, attachment = next(iter(bonds))
            yield anchor, attachment, frozenset(component)


def substituents(mol: Chem.Mol, core: frozenset[int]) -> dict[int, Substituent | None]:
    """Return the single heavy substituent hanging off each core atom, or ``None``.

    A core atom carrying two heavy branches is omitted from the result entirely rather
    than reported as either of them: replacing "the" substituent there is ambiguous, and
    an ambiguous edit is not one a reviewer can check.
    """
    heavy: dict[int, list[Substituent]] = {}
    for anchor, attachment, component in branches(mol, core):
        heavy_atoms = [idx for idx in component if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1]
        if not heavy_atoms:
            continue
        heavy.setdefault(anchor, []).append(
            Substituent(
                attachment=attachment,
                atoms=component,
                heavy=len(heavy_atoms),
                key=Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(component), canonical=True),
            )
        )
    result: dict[int, Substituent | None] = {}
    for anchor in core:
        found = heavy.get(anchor, [])
        if len(found) > 1:
            continue
        result[anchor] = found[0] if found else None
    return result


def first_hydrogen(mol: Chem.Mol, idx: int) -> int | None:
    """Index of an explicit hydrogen on atom *idx*, or ``None``."""
    for neighbour in mol.GetAtomWithIdx(idx).GetNeighbors():
        if neighbour.GetAtomicNum() == 1:
            return neighbour.GetIdx()
    return None


def shared_core(
    source: Ligand, target: Ligand, mapping_options: MappingOptions, trace: list[str]
) -> dict[int, int] | None:
    """Return the shared-core correspondence, resolving symmetry by geometry.

    Parameters
    ----------
    source, target : Ligand
        Co-posed parents.
    mapping_options : MappingOptions
        The same settings the pipeline's mappers run under, so a generator finds the core
        the mapper would have found.
    trace : list of str
        Appended to, so the choice is visible in the proposal's trace.

    Returns
    -------
    dict[int, int] or None
        ``{source atom: target atom}``, or ``None`` when no common core exists.
    """
    pattern = mcs_query(source.mol, target.mol, mapping_options, match_elements=True)
    if pattern is None:
        return None
    here, there = mcs_embeddings(source.mol, target.mol, pattern, mapping_options)
    if not here or not there:
        return None
    source_coords = np.asarray(source.mol.GetConformer().GetPositions(), dtype=float)
    target_coords = np.asarray(target.mol.GetConformer().GetPositions(), dtype=float)
    best, best_rmsd = None, float("inf")
    for match_here in here:
        for match_there in there:
            rmsd = core_rmsd(source_coords[list(match_here)], target_coords[list(match_there)])
            if rmsd < best_rmsd:
                best, best_rmsd = dict(zip(match_here, match_there)), rmsd
    trace.append(f"core of {len(best or ())} atom(s) chosen at in-place RMSD {best_rmsd:.3f} A")
    return best


def differing_positions(source: Ligand, target: Ligand, core: Mapping[int, int]) -> list[Position]:
    """Return the core positions where the two parents carry different groups.

    Parameters
    ----------
    source, target : Ligand
    core : Mapping[int, int]
        The correspondence from :func:`shared_core`.

    Returns
    -------
    list[Position]
        In increasing source-atom order, so a generator's output does not depend on
        dictionary iteration order.
    """
    source_subs = substituents(source.mol, frozenset(core))
    target_subs = substituents(target.mol, frozenset(core.values()))
    positions: list[Position] = []
    for s_idx, t_idx in sorted(core.items()):
        if s_idx not in source_subs or t_idx not in target_subs:
            continue
        here, over_there = source_subs[s_idx], target_subs[t_idx]
        key_here = HYDROGEN_KEY if here is None else here.key
        key_there = HYDROGEN_KEY if over_there is None else over_there.key
        if key_here == key_there:
            continue
        positions.append(Position(source_anchor=s_idx, target_anchor=t_idx, source=here, target=over_there))
    return positions


def assemble(
    source: Ligand, target: Ligand, positions: Sequence[Position], choices: Sequence[str]
) -> tuple[Chem.Mol, dict[str, dict[int, int]]] | None:
    """Build the molecule that puts *choices* at *positions*, with its parent atom maps.

    Parameters
    ----------
    source, target : Ligand
        The parents. The result is the source with some of its groups replaced.
    positions : Sequence[Position]
        The differing positions, from :func:`differing_positions`.
    choices : Sequence[str]
        One of ``"source"``, ``"target"``, ``"hydrogen"`` per position, in the same order.

    Returns
    -------
    tuple[rdkit.Chem.Mol, dict[str, dict[int, int]]] or None
        The molecule -- conformer-free -- and ``{parent name: {new index: parent index}}``.
        ``None`` when the result does not sanitize, which a caller records in a trace
        rather than treating as an error.

    Notes
    -----
    Built by combining both parents, adding the new bonds, and deleting what is not
    wanted, so **every surviving atom's origin is known exactly**. That is the whole
    reason to build it this way rather than from SMILES: the resulting
    ``parent_atom_map`` spares the poser a substructure search whose symmetry it would
    otherwise have to resolve by guessing, which is the same coin-flip
    :func:`shared_core` exists to remove.

    Hydrogens are left implicit at a truncated position rather than added here.
    :func:`~rbfenetmap.core.posing.pose_intermediate` calls ``AddHs`` before embedding --
    hydrogens placed by rule before a geometry exists are a drawing, not a pose -- and
    ``AddHs`` appends, so every index in the returned maps stays valid.
    """
    n_source = source.mol.GetNumAtoms()
    combined = Chem.RWMol(Chem.CombineMols(source.mol, target.mol))

    drop: set[int] = set()
    keep_target: set[int] = set()
    new_bonds: list[tuple[int, int, Chem.BondType]] = []
    touched: set[int] = set()

    for position, choice in zip(positions, choices):
        if choice == "source":
            continue
        touched.add(position.source_anchor)
        if position.source is not None:
            drop |= set(position.source.atoms)
        elif choice == "target":
            hydrogen = first_hydrogen(source.mol, position.source_anchor)
            if hydrogen is None:
                return None
            drop.add(hydrogen)
        if choice == "target" and position.target is not None:
            keep_target |= set(position.target.atoms)
            bond = target.mol.GetBondBetweenAtoms(position.target_anchor, position.target.attachment)
            new_bonds.append((position.source_anchor, n_source + position.target.attachment, bond.GetBondType()))

    drop |= {n_source + idx for idx in range(target.mol.GetNumAtoms()) if idx not in keep_target}

    for anchor in touched:
        atom = combined.GetAtomWithIdx(anchor)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    for begin, end, order in new_bonds:
        combined.AddBond(begin, end, order)

    for idx in sorted(drop, reverse=True):
        combined.RemoveAtom(idx)

    keep = [idx for idx in range(n_source + target.mol.GetNumAtoms()) if idx not in drop]
    renumber = {old: new for new, old in enumerate(keep)}

    mol = combined.GetMol()
    mol.RemoveAllConformers()
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - an unbuildable hybrid is data, not an error
        return None

    atom_map = {
        source.name: {renumber[old]: old for old in keep if old < n_source},
        target.name: {renumber[old]: old - n_source for old in keep if old >= n_source},
    }
    return mol, atom_map
