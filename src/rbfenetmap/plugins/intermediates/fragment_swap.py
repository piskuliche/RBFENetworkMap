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

The decomposition and the molecule construction both live in
:mod:`rbfenetmap.plugins.intermediates._rgroups`, shared with the PairMap generator. They
were factored out rather than copied: the decomposition decides which atoms count as "the
same position" on the two parents, and two copies that drifted would mean the two
generators disagreeing about what a molecule *is* while both looking correct in isolation.

The molecule is built by combining both parents, adding the one new bond, and deleting
what is not wanted -- which means every surviving atom's origin is known exactly. That is
what lets the generator hand over a complete ``parent_atom_map`` and spare the poser a
substructure search whose symmetry it would have to resolve by guessing.
"""

from __future__ import annotations

from typing import ClassVar, Mapping

from rbfenetmap.core.intermediates import (
    IntermediateOptions,
    IntermediateProposal,
    ProposedLink,
    ProposedMolecule,
    intermediate_name,
)
from rbfenetmap.core.meta.intermediates import AbstractIntermediateGenerator
from rbfenetmap.core.models import Ligand
from rbfenetmap.core.options import MappingOptions
from rbfenetmap.plugins.intermediates._rgroups import assemble, differing_positions, shared_core

__all__ = ("FragmentSwapGenerator",)


class FragmentSwapGenerator(AbstractIntermediateGenerator):
    """Propose the hybrids that swap one substituent at a time.

    Notes
    -----
    Rejects with ``"single_substituent_difference"`` when the parents differ at only one
    position. That is not a limitation to be worked around: with one difference, the only
    hybrids are the parents themselves, so there is genuinely no intermediate to invent
    and a generator that returned one would be returning a duplicate ligand. A generator
    that can do something useful there --
    :class:`~rbfenetmap.plugins.intermediates.pairmap_generator.PairMapGenerator`, by
    truncating the position to the shared core -- is a different generator.
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
            ``max_molecules`` caps how many hybrids are returned. The subnetwork knobs are
            not consulted: this generator emits a fan of independent two-link paths, not a
            searched subnetwork.
        mapping_options : MappingOptions
            Settings for the MCS that finds the shared core.

        Returns
        -------
        IntermediateProposal
        """
        trace: list[str] = []
        core = shared_core(source, target, mapping_options, trace)
        if core is None:
            return IntermediateProposal(
                source=source.name,
                target=target.name,
                generator=self.name,
                rejection="no_common_core",
                trace=tuple(trace),
            )

        positions = differing_positions(source, target, core)
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
        for index, position in enumerate(positions):
            if len(molecules) >= options.max_molecules:
                trace.append(f"stopped at max_molecules={options.max_molecules}")
                break
            choices = ["target" if other == index else "source" for other in range(len(positions))]
            built = assemble(source, target, positions, choices)
            if built is None:
                trace.append(f"position {position.source_anchor}->{position.target_anchor}: could not be built")
                continue
            mol, atom_map = built
            proposed = ProposedMolecule(
                mol=mol,
                parents=(source.name, target.name),
                parent_atom_map=atom_map,
                detail={"swapped_position": position.source_anchor},
            )
            molecules.append(proposed)
            invented = intermediate_name(proposed.parents, proposed.mol)
            links.append(ProposedLink(source=source.name, target=invented))
            links.append(ProposedLink(source=invented, target=target.name))
            trace.append(f"position {position.source_anchor}->{position.target_anchor}: proposed {invented}")

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
