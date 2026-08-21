"""MCS-based atom mappers.

Ports ``BuildEdges._find_mcs`` and the ``MCSS`` / ``MCSS-E`` / ``MCSS-E2`` family, with
one substantive change: symmetric substructures are resolved explicitly instead of by
whichever embedding RDKit happens to return first. See :meth:`MCSSMapper.map_pair`.
"""

from __future__ import annotations

from itertools import permutations
from typing import ClassVar, Sequence

import numpy as np

from rbfenetmap.core.coreprune import prune_core
from rbfenetmap.core.exceptions import MappingError
from rbfenetmap.core.kabsch import core_rmsd
from rbfenetmap.core.mcs import mcs_embeddings, mcs_query
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.models import AtomMapping, Ligand
from rbfenetmap.core.molgraph import connected_components_of, hydrogen_parents, mol_to_graph
from rbfenetmap.core.options import CorePruningPolicy, MappingOptions

__all__ = ("MCSSExtended2Mapper", "MCSSExtendedMapper", "MCSSMapper")


def _pair_hydrogens(source: Ligand, target: Ligand, core: dict[int, int]) -> dict[int, int]:
    """Extend a heavy-atom correspondence to the hydrogens hanging off it.

    The MCS runs on the heavy-atom graph -- see
    :func:`~rbfenetmap.core.mcs._suppress_hydrogens` for why -- so *core* pairs only heavy
    atoms. The hydrogens have to be paired here, and they cannot simply be left out.

    Leaving them out is not the harmless-looking omission it appears to be. Every unpaired
    hydrogen is soft-core, so a molecule's hydrogens would arrive as dozens of one-atom
    soft-core regions on atoms whose parents are all common core. The repair's
    hydrogen-follows-parent rule is deliberately one-way and would not reclaim them, so the
    Steiner search would set about bridging them and demote the entire core doing it.

    Pairing is by parent, and *within* a parent by geometry: the assignment minimising total
    squared displacement, ties broken on index so the result is reproducible. Hydrogens on
    one heavy atom are chemically interchangeable, which makes any assignment equally valid
    as chemistry -- but not as geometry. Pairing them in index order instead costs a mean
    0.17 A and up to 0.61 A of ``core_rmsd`` on the example benzamides, and ``core_rmsd``
    both gates edges at ``core_rmsd_threshold`` and carries weight in the scorer. The
    permutation search is bounded by the four hydrogens a heavy atom can carry.

    Where the two parents carry different numbers of hydrogens, the surplus stays soft-core,
    and that is what carries the element-changing cases now that the MCS never sees a
    hydrogen. In ``R-H -> R-Cl`` the two carbons pair, the chlorine has no heavy counterpart
    and is soft-core, and the hydrogen finds nothing to pair with because its partner carbon
    carries none -- so it is soft-core too, which is the correct description of that
    transformation. Previously the same outcome came about by a different route: a permissive
    MCS paired the hydrogen *with* the chlorine and
    :func:`~rbfenetmap.core.coreprune.prune_core` demoted it through
    ``demote_light_element_swap``. That path is now unreachable, so this one is load-bearing.
    """
    parents_1: dict[int, list[int]] = {}
    for hydrogen, parent in hydrogen_parents(source.mol).items():
        parents_1.setdefault(parent, []).append(hydrogen)
    parents_2: dict[int, list[int]] = {}
    for hydrogen, parent in hydrogen_parents(target.mol).items():
        parents_2.setdefault(parent, []).append(hydrogen)
    if not parents_1 or not parents_2:
        return dict(core)

    coords_1 = np.asarray(source.mol.GetConformer().GetPositions(), dtype=float)
    coords_2 = np.asarray(target.mol.GetConformer().GetPositions(), dtype=float)

    extended = dict(core)
    for parent_1, parent_2 in core.items():
        hydrogens_1 = sorted(parents_1.get(parent_1, ()))
        hydrogens_2 = sorted(parents_2.get(parent_2, ()))
        if not hydrogens_1 or not hydrogens_2:
            continue
        # Permute the *longer* list against the shorter, so which hydrogens make up the
        # surplus is also chosen by geometry rather than by whichever sorted last.
        if len(hydrogens_1) <= len(hydrogens_2):
            fixed, fixed_coords, pool, pool_coords, flipped = hydrogens_1, coords_1, hydrogens_2, coords_2, False
        else:
            fixed, fixed_coords, pool, pool_coords, flipped = hydrogens_2, coords_2, hydrogens_1, coords_1, True
        best: tuple[float, tuple[int, ...]] | None = None
        for candidate in permutations(pool, len(fixed)):
            cost = float(np.sum((fixed_coords[fixed] - pool_coords[list(candidate)]) ** 2))
            key = (cost, candidate)
            if best is None or key < best:
                best = key
        if best is None:  # pragma: no cover - a non-empty pool always yields a permutation
            continue
        for anchor, partner in zip(fixed, best[1]):
            if flipped:
                extended[partner] = anchor
            else:
                extended[anchor] = partner
    return extended


class MCSSMapper(AbstractMapper):
    """Map two ligands by their maximum common substructure.

    The base mapper applies no property-based core pruning; the subclasses below enable
    the degree and element rules that reproduce ``MCSS-E`` and ``MCSS-E2``.
    """

    name: ClassVar[str] = "mcss"
    #: Overridden by subclasses to enable pruning rules on top of the caller's options.
    pruning_preset: ClassVar[str | None] = "mcss"

    def map_pair(self, source: Ligand, target: Ligand, options: MappingOptions) -> AtomMapping:
        """Return the MCS correspondence between *source* and *target*.

        Parameters
        ----------
        source, target : Ligand
        options : MappingOptions
            ``timeout``, the ``FindMCS`` ring settings, ``max_matches``, and
            ``match_selection``.

        Returns
        -------
        AtomMapping

        Raises
        ------
        rbfenetmap.core.exceptions.MappingError
            If no common substructure exists, or the SMARTS it produces cannot be
            matched back onto either molecule.

        Notes
        -----
        The correspondence is built by enumerating substructure *embeddings* of the MCS
        SMARTS in both molecules and choosing a pairing deliberately, rather than by
        calling the singular ``GetSubstructMatch`` on each molecule and zipping the
        results as ``BuildEdges._find_mcs`` does.

        That zip is a coin flip for any symmetric substructure. A *para*-substituted
        benzene has two embeddings of its ring related by a flip; if RDKit returns
        different ones for the two molecules, the resulting map pairs atoms across the
        ring from one another. The mapping is topologically valid, so nothing detects the
        problem until a geometry check much later -- by which point the failure looks like
        a bad conformer rather than a bad correspondence.

        Candidate pairings are ranked by the criterion in ``match_selection``:
        ``"fewest_fragments"`` (default) prefers the pairing whose soft-core is least
        fragmented, which directly reduces the work the repair has to do;
        ``"best_rmsd"`` prefers the geometrically closest; ``"first"`` restores the old
        behaviour for comparison.
        """
        pattern = mcs_query(source.mol, target.mol, options)
        if pattern is None:
            raise MappingError(
                f"No usable common substructure between {source.name!r} and {target.name!r}. "
                "Either the molecules share nothing, or the MCS SMARTS could not be parsed "
                "back into a query molecule."
            )

        matches_1, matches_2 = mcs_embeddings(source.mol, target.mol, pattern, options)
        if not matches_1 or not matches_2:
            raise MappingError(
                f"{source.name}~{target.name}: MCS found but its SMARTS does not match "
                f"{'source' if not matches_1 else 'target'}."
            )

        core = self._select_pairing(source, target, matches_1, matches_2, options)
        core = _pair_hydrogens(source, target, core)
        mapping = AtomMapping.from_core_pairs(
            core, n_atoms_1=source.n_atoms, n_atoms_2=target.n_atoms, method=self.name
        )
        return prune_core(source, target, mapping, self._pruning_policy(options))

    def _pruning_policy(self, options: MappingOptions) -> CorePruningPolicy:
        """Combine the caller's policy with this mapper's preset."""
        if self.pruning_preset is None:
            return options.core_pruning
        preset = CorePruningPolicy.preset(self.pruning_preset)
        # The caller's charge and light-element settings are preferences, not part of the
        # historical method definition, so they survive the preset.
        return CorePruningPolicy(
            demote_element_mismatch=preset.demote_element_mismatch,
            demote_degree_mismatch=preset.demote_degree_mismatch,
            demote_formal_charge_mismatch=options.core_pruning.demote_formal_charge_mismatch,
            demote_aromaticity_mismatch=options.core_pruning.demote_aromaticity_mismatch,
            demote_ring_membership_mismatch=options.core_pruning.demote_ring_membership_mismatch,
            demote_light_element_swap=options.core_pruning.demote_light_element_swap,
        )

    def _select_pairing(
        self,
        source: Ligand,
        target: Ligand,
        matches_1: Sequence[Sequence[int]],
        matches_2: Sequence[Sequence[int]],
        options: MappingOptions,
    ) -> dict[int, int]:
        """Choose one embedding pairing according to ``options.match_selection``."""
        if options.match_selection == "first":
            return dict(zip(matches_1[0], matches_2[0]))

        graph_1 = mol_to_graph(source.mol)
        graph_2 = mol_to_graph(target.mol)
        all_1 = set(range(source.n_atoms))
        all_2 = set(range(target.n_atoms))
        # Both conformers are fixed for the whole search, so extract each one's full
        # coordinate matrix once and index it per candidate rather than rebuilding the
        # arrays atom by atom inside the loop below. `GetPositions` returns the whole
        # (n, 3) block in one call, and the loop runs up to `max_matches` times.
        coords_1 = np.asarray(source.mol.GetConformer().GetPositions(), dtype=float)
        coords_2 = np.asarray(target.mol.GetConformer().GetPositions(), dtype=float)

        # What matters is the *relative* orientation of the two embeddings, and the set
        # of achievable relative orientations is the cross product -- varying only one
        # side is not enough. Fixing the source and varying the target explores nothing
        # when the symmetry lives in the source: a benzamide whose substituent is a plain
        # hydrogen has a symmetric ring, while its bulkier partner does not, so the
        # target contributes a single embedding and the ring flip is never considered.
        # The product is bounded by `max_matches` so a highly symmetric pair cannot blow
        # up the search.
        budget = max(options.max_matches, 1)
        best: tuple[tuple[float, ...], dict[int, int]] | None = None
        evaluated = 0
        for match_1 in matches_1:
            for match_2 in matches_2:
                if evaluated >= budget:
                    break
                evaluated += 1
                core = dict(zip(match_1, match_2))
                if len(set(core.values())) != len(core):
                    continue
                fragments = len(connected_components_of(graph_1, all_1 - set(core))) + len(
                    connected_components_of(graph_2, all_2 - set(core.values()))
                )
                rmsd = core_rmsd(coords_1[list(core)], coords_2[list(core.values())])
                key = (
                    (float(fragments), rmsd)
                    if options.match_selection == "fewest_fragments"
                    else (rmsd, float(fragments))
                )
                if best is None or key < best[0]:
                    best = (key, core)
            if evaluated >= budget:
                break

        if best is None:  # pragma: no cover - defensive; a valid MCS always yields one
            return dict(zip(matches_1[0], matches_2[0]))
        return best[1]


class MCSSExtendedMapper(MCSSMapper):
    """MCS mapping that additionally demotes pairs whose connectivity differs.

    Reproduces ``BuildEdges``' ``MCSS-E``.
    """

    name: ClassVar[str] = "mcss-e"
    pruning_preset: ClassVar[str | None] = "mcss-e"


class MCSSExtended2Mapper(MCSSMapper):
    """MCS mapping that demotes pairs differing in element or connectivity.

    Reproduces ``BuildEdges``' ``MCSS-E2``. The default mapper for the CLI: holding a
    carbon fixed against a nitrogen is legal topologically but rarely what anyone wants
    from a free energy calculation.
    """

    name: ClassVar[str] = "mcss-e2"
    pruning_preset: ClassVar[str | None] = "mcss-e2"
