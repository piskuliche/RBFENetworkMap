"""One-substituent-at-a-time intermediate generator.

Deliberately the simplest thing that is genuinely an intermediate generator, and it plays
the role :class:`~rbfenetmap.plugins.mappers.identity_mapper.IdentityMapper` plays for
mappers: it exercises the whole seam -- proposal, atom map, posing, naming, provenance --
without any chemistry that can surprise a reviewer.

The idea
--------

Take the common core of the two ligands. Where they differ, they differ at a handful of
substituent positions. If they differ at only one, there is nothing to invent: any hybrid
*is* one of the parents. If they differ at two or more, then for each differing position
there is a molecule that is the source with exactly that one substituent replaced by the
target's -- a molecule strictly closer to the target than the source is, and strictly
closer to the source than the target is. That is the entire algorithm.

What it deliberately does not do
--------------------------------

No scaffold hops, no ring transformations, no linker growth, no search over combinations
of positions. A real generator (PairMap) chooses which of the many possible hybrids are
worth the compute; this one enumerates the single-swap ones in a fixed order and stops at
the budget. Its value is that its output is obvious by inspection, so a failure anywhere
downstream is unambiguously downstream.

Bookkeeping
-----------

The molecule is built by combining both parents, adding the one new bond, and deleting
what is not wanted -- which means every surviving atom's origin is known exactly. That is
what lets the generator hand over a complete ``parent_atom_map`` and spare the poser a
substructure search whose symmetry it would have to resolve by guessing.
"""

from __future__ import annotations

from typing import ClassVar, Iterator, Mapping

import numpy as np
from rdkit import Chem

from rbfenetmap.core.intermediates import (
    IntermediateOptions,
    IntermediateProposal,
    ProposedLink,
    ProposedMolecule,
    intermediate_name,
)
from rbfenetmap.core.kabsch import core_rmsd
from rbfenetmap.core.mcs import mcs_embeddings, mcs_query
from rbfenetmap.core.meta.intermediates import AbstractIntermediateGenerator
from rbfenetmap.core.models import Ligand
from rbfenetmap.core.options import MappingOptions

__all__ = ("FragmentSwapGenerator",)

#: Key used for a position carrying nothing but hydrogen, so an ``H -> CH3`` position
#: compares as a difference like any other.
_HYDROGEN_KEY = "[H]"


def _branches(mol: Chem.Mol, core: frozenset[int]) -> Iterator[tuple[int, int, frozenset[int]]]:
    """Yield ``(core atom, attachment atom, component)`` for each simple substituent.

    A component attached to the core at more than one bond -- a fused ring, a macrocyclic
    linker -- is skipped rather than reported. Swapping one is not a substituent swap, and
    handling it is exactly the chemistry this generator exists to avoid.
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


def _substituents(mol: Chem.Mol, core: frozenset[int]) -> dict[int, tuple[int, frozenset[int]] | None]:
    """Return the single heavy substituent hanging off each core atom, or ``None``.

    ``None`` means the position carries only hydrogens, which is a substituent like any
    other as far as a swap is concerned. A position carrying *two* heavy branches is
    omitted from the result entirely: replacing "the" substituent there is ambiguous, and
    an ambiguous edit is not a trivial one.
    """
    heavy: dict[int, list[tuple[int, frozenset[int]]]] = {}
    for anchor, attachment, component in _branches(mol, core):
        if any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 1 for idx in component):
            heavy.setdefault(anchor, []).append((attachment, component))
    result: dict[int, tuple[int, frozenset[int]] | None] = {}
    for anchor in core:
        branches = heavy.get(anchor, [])
        if len(branches) > 1:
            continue
        result[anchor] = branches[0] if branches else None
    return result


def _substituent_key(mol: Chem.Mol, substituent: tuple[int, frozenset[int]] | None) -> str:
    """Canonical SMILES of a substituent, or :data:`_HYDROGEN_KEY` for a bare position."""
    if substituent is None:
        return _HYDROGEN_KEY
    _, component = substituent
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(component), canonical=True)


def _first_hydrogen(mol: Chem.Mol, idx: int) -> int | None:
    """Index of an explicit hydrogen on atom *idx*, or ``None``."""
    for neighbour in mol.GetAtomWithIdx(idx).GetNeighbors():
        if neighbour.GetAtomicNum() == 1:
            return neighbour.GetIdx()
    return None


class FragmentSwapGenerator(AbstractIntermediateGenerator):
    """Propose the hybrids that swap one substituent at a time.

    Notes
    -----
    Rejects with ``"single_substituent_difference"`` when the parents differ at only one
    position. That is not a limitation to be worked around: with one difference, the only
    hybrids are the parents themselves, so there is genuinely no intermediate to invent
    and a generator that returned one would be returning a duplicate ligand.
    """

    name: ClassVar[str] = "fragment-swap"

    def propose(
        self, source: Ligand, target: Ligand, options: IntermediateOptions, mapping_options: MappingOptions
    ) -> IntermediateProposal:
        """Return one hybrid per differing substituent position.

        Parameters
        ----------
        source, target : Ligand
            The gap endpoints.
        options : IntermediateOptions
            ``max_molecules`` caps how many hybrids are returned.
        mapping_options : MappingOptions
            Settings for the MCS that finds the shared core.

        Returns
        -------
        IntermediateProposal
        """
        trace: list[str] = []
        core = self._core(source, target, mapping_options, trace)
        if core is None:
            return IntermediateProposal(
                source=source.name,
                target=target.name,
                generator=self.name,
                rejection="no_common_core",
                trace=tuple(trace),
            )

        source_subs = _substituents(source.mol, frozenset(core))
        target_subs = _substituents(target.mol, frozenset(core.values()))
        positions = [
            (s_idx, t_idx)
            for s_idx, t_idx in sorted(core.items())
            if s_idx in source_subs
            and t_idx in target_subs
            and _substituent_key(source.mol, source_subs[s_idx]) != _substituent_key(target.mol, target_subs[t_idx])
        ]
        trace.append(f"{len(positions)} differing substituent position(s) on a {len(core)}-atom core")
        if len(positions) < 2:
            return IntermediateProposal(
                source=source.name,
                target=target.name,
                generator=self.name,
                rejection="single_substituent_difference",
                trace=tuple(trace),
            )

        molecules: list[ProposedMolecule] = []
        links: list[ProposedLink] = []
        for s_idx, t_idx in positions:
            if len(molecules) >= options.max_molecules:
                trace.append(f"stopped at max_molecules={options.max_molecules}")
                break
            built = self._swap(source, target, s_idx, t_idx, source_subs[s_idx], target_subs[t_idx])
            if built is None:
                trace.append(f"position {s_idx}->{t_idx}: could not be built")
                continue
            mol, atom_map = built
            proposed = ProposedMolecule(
                mol=mol,
                parents=(source.name, target.name),
                parent_atom_map=atom_map,
                detail={"swapped_position": s_idx},
            )
            molecules.append(proposed)
            invented = intermediate_name(proposed.parents, proposed.mol)
            links.append(ProposedLink(source=source.name, target=invented))
            links.append(ProposedLink(source=invented, target=target.name))
            trace.append(f"position {s_idx}->{t_idx}: proposed {invented}")

        if not molecules:
            return IntermediateProposal(
                source=source.name,
                target=target.name,
                generator=self.name,
                rejection="no_valid_hybrid",
                trace=tuple(trace),
            )
        return IntermediateProposal(
            source=source.name,
            target=target.name,
            generator=self.name,
            molecules=tuple(molecules),
            links=tuple(links),
            trace=tuple(trace),
        )

    def describe_parameters(self) -> Mapping[str, object]:
        """Return the generator's settings. It has none of its own."""
        return {"swaps_per_molecule": 1}

    # -- internals ----------------------------------------------------------------

    @staticmethod
    def _core(
        source: Ligand, target: Ligand, mapping_options: MappingOptions, trace: list[str]
    ) -> dict[int, int] | None:
        """Return the shared-core correspondence, resolving symmetry by geometry.

        ``match_elements=True`` and the embedding chosen by lowest in-place RMSD, for the
        same reason the poser does both: here the correspondence decides which atoms are
        "the same position" on the two parents, and getting that wrong swaps a substituent
        onto the far side of a ring while looking entirely reasonable on paper.
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

    @staticmethod
    def _swap(
        source: Ligand,
        target: Ligand,
        s_idx: int,
        t_idx: int,
        source_sub: tuple[int, frozenset[int]] | None,
        target_sub: tuple[int, frozenset[int]] | None,
    ) -> tuple[Chem.Mol, dict[str, dict[int, int]]] | None:
        """Build the source with position *s_idx* carrying the target's substituent.

        Returns ``None`` when the result does not sanitize, which a caller records in the
        trace rather than treating as an error.
        """
        n_source = source.mol.GetNumAtoms()
        combined = Chem.RWMol(Chem.CombineMols(source.mol, target.mol))

        # Atoms to drop: the whole source branch (or one hydrogen when the position is
        # bare), and everything of the target except the branch being transplanted.
        drop: set[int] = set()
        if source_sub is not None:
            drop |= set(source_sub[1])
        else:
            hydrogen = _first_hydrogen(source.mol, s_idx)
            if hydrogen is None:
                return None
            drop.add(hydrogen)

        keep_target: frozenset[int] = frozenset() if target_sub is None else target_sub[1]
        drop |= {n_source + idx for idx in range(target.mol.GetNumAtoms()) if idx not in keep_target}

        if target_sub is not None:
            attachment = n_source + target_sub[0]
            order = target.mol.GetBondBetweenAtoms(t_idx, target_sub[0]).GetBondType()
            combined.AddBond(s_idx, attachment, order)

        anchor = combined.GetAtomWithIdx(s_idx)
        anchor.SetNoImplicit(False)
        anchor.SetNumExplicitHs(0)

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
