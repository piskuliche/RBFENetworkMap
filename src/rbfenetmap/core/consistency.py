"""Graph-wide core consistency: one core per ligand, not one per edge.

The default, ``consistency="pairwise"``, maps each edge independently. A ligand sitting on
three edges therefore holds three different common cores, one per partner, and nothing
requires them to agree. That is the right default -- each edge gets the largest core its own
pair supports, which is the cheapest transformation for that pair -- but it means the series
has no single shared scaffold. Whether an atom is "in the core" is a question that can only
be answered per edge.

``consistency="graph"`` answers it per *ligand*. Each ligand keeps the intersection of the
cores it holds across all of its selected RBFE edges; everything else is demoted to
soft-core and the repair is re-run on what remains. The result is a genuine common core for
the whole (connected) network rather than a merely pairwise-compatible one, which is what
makes a group of ligands share a scaffold in the sense a per-cluster Amber setup wants.

Why it iterates
---------------
Intersecting is not a single pass. Demoting an atom on one side drops its partner on the
other, which shrinks that ligand's core, which changes *its* intersection; and the soft-core
repair may demote further atoms still to keep the soft-core in one connected piece. Cores
only ever shrink, so the iteration is monotone in a finite set and terminates -- it is run
to a fixed point rather than applied once.

What it does not do
-------------------
It does not re-select. Shrinking a core makes an edge dearer, and in principle a different
network would be optimal under the reduced cores; recomputing selection here would mean
re-mapping the whole candidate pool under a constraint that depends on which edges were
selected, which is circular. This is a post-selection refinement of the mappings of the
edges that were chosen, and the costs it recomputes are reported honestly rather than fed
back into selection.

CBFE edges are ignored. A counterpoised edge has no common core by construction, so reading
one as "this ligand's core is empty here" would erase the core of every ligand a bridge
touches -- an artefact of the bridge, not a statement about the scaffold.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Mapping, Sequence

from rbfenetmap.core.descriptors import compute_descriptors
from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.meta.scorers import AbstractScorer
from rbfenetmap.core.models import AtomMapping, EdgeKind, Ligand, Network, RejectionReason, Transformation
from rbfenetmap.core.options import NetworkOptions, SoftcorePolicy
from rbfenetmap.core.softcore import repair_softcore_connectivity

__all__ = ("apply_graph_consistency", "graph_consistent_cores", "maybe_apply_graph_consistency")

logger = logging.getLogger(__name__)


def _core_by_ligand(edges: Sequence[Transformation]) -> dict[str, set[int]]:
    """Intersect each ligand's common core over the RBFE edges incident to it."""
    cores: dict[str, set[int]] = {}
    for edge in edges:
        for name, atoms in ((edge.source, set(edge.mapping.cc1)), (edge.target, set(edge.mapping.cc2))):
            current = cores.get(name)
            cores[name] = atoms if current is None else current & atoms
    return cores


def _restrict(mapping: AtomMapping, keep_1: set[int], keep_2: set[int]) -> AtomMapping:
    """Return *mapping* with every core pair not surviving on **both** sides demoted.

    A pair is kept only if each of its two atoms is in its own ligand's surviving set. An
    atom whose partner was dropped has nothing left to correspond to, so it becomes
    soft-core too -- which is exactly why one pass is not enough.
    """
    pairs = [(a, b) for a, b in zip(mapping.cc1, mapping.cc2) if a in keep_1 and b in keep_2]
    return AtomMapping.from_core_pairs(
        pairs, n_atoms_1=mapping.n_atoms_1, n_atoms_2=mapping.n_atoms_2, method=mapping.method
    )


def graph_consistent_cores(network: Network, *, policy: SoftcorePolicy | None = None) -> dict[str, frozenset[int]]:
    """Return the atoms each ligand keeps in the graph-consistent core.

    Parameters
    ----------
    network : Network
        A planned network. Only its selected RBFE edges are read.
    policy : SoftcorePolicy, optional
        Used for the repair run between intersection passes. Defaults to the network's own
        policy, or to library defaults if the network carries no options.

    Returns
    -------
    dict[str, frozenset[int]]
        Atom indices, per ligand name. A ligand with no selected RBFE edge is absent: it is
        under no constraint, because consistency is a statement about atoms shared *across*
        edges and it has none.

    Notes
    -----
    Exposed separately from :func:`apply_graph_consistency` because the surviving core is
    the answer to "do these ligands share a scaffold at all, and how big is it?", which is
    worth asking without rewriting a network to find out.
    """
    return {name: frozenset(atoms) for name, atoms in _fixed_point(network, policy)[0].items()}


def _fixed_point(
    network: Network, policy: SoftcorePolicy | None
) -> tuple[dict[str, set[int]], dict[tuple[str, str], AtomMapping]]:
    """Iterate intersect-then-repair until no core shrinks further.

    Returns the per-ligand surviving cores and the per-pair repaired mappings. Termination
    is guaranteed because every pass is non-increasing on a finite set of atoms and the loop
    exits as soon as a pass changes nothing.
    """
    policy = policy or (network.options.softcore if network.options is not None else SoftcorePolicy())
    rbfe = [edge for edge in network.edges if edge.kind is EdgeKind.RBFE]
    mappings = {edge.unordered_key: edge for edge in rbfe}

    current: dict[tuple[str, str], AtomMapping] = {pair: edge.mapping for pair, edge in mappings.items()}
    # Bounded by the total core size, which strictly decreases on every pass that is not the
    # last; the +2 covers the confirming pass and the degenerate empty-network case.
    limit = sum(len(mapping.cc1) for mapping in current.values()) + 2
    cores: dict[str, set[int]] = {}
    for _ in range(limit):
        oriented = [replace(mappings[pair], mapping=current[pair]) for pair in sorted(current)]
        cores = _core_by_ligand(oriented)
        changed = False
        for pair, edge in sorted(mappings.items()):
            mapping = current[pair]
            keep_1 = cores.get(edge.source, set(mapping.cc1))
            keep_2 = cores.get(edge.target, set(mapping.cc2))
            restricted = _restrict(mapping, keep_1, keep_2)
            if len(restricted.cc1) == len(mapping.cc1):
                continue
            source, target = network.ligands[edge.source], network.ligands[edge.target]
            repaired, _ = repair_softcore_connectivity(source, target, restricted, policy)
            current[pair] = repaired
            changed = True
        if not changed:
            break

    return cores, current


def apply_graph_consistency(
    network: Network, *, scorer: AbstractScorer | str = "linear", policy: SoftcorePolicy | None = None
) -> Network:
    """Rewrite *network*'s selected edges onto one core per ligand.

    Parameters
    ----------
    network : Network
        A planned network. Its candidate pool, planner, and edge *selection* are unchanged;
        only the mappings, repairs, and costs of the selected RBFE edges are rewritten.
    scorer : AbstractScorer or str, optional
        Used to re-cost the reduced edges. A string is looked up in the scorer registry.
        Pass the same scorer the network was planned with -- costs computed by two different
        scorers are not comparable, and the returned network holds a mixture of neither.
    policy : SoftcorePolicy, optional
        Feasibility policy for the re-run repair. Defaults to the network's own.

    Returns
    -------
    Network
        With graph-consistent mappings and recomputed costs.

    Raises
    ------
    rbfenetmap.core.exceptions.NetworkPlanError
        If any selected edge becomes infeasible under the reduced core.

    Notes
    -----
    The failure mode is a hard error rather than a per-edge rejection, and that is the one
    design decision here worth arguing about. Elsewhere in the package an infeasible edge is
    recorded and kept, because it is a *candidate* nobody has to run. These are selected
    edges: a network handed back containing an edge marked infeasible is a network that
    cannot be run, and quietly reverting the offending edges to their pairwise cores would
    hand back something that is not graph-consistent while claiming to be -- the exact
    failure ``--consistency graph`` was reported for in the first place.

    A raise here also carries real information: it means these ligands do not share a core
    large enough to run on, which is a fact about the series, and the message names the
    edges and reasons so the user can loosen a threshold, drop a ligand, or plan the
    subsets separately.
    """
    from rbfenetmap.plugins.scorers import create_scorer

    scorer_obj = create_scorer(scorer) if isinstance(scorer, str) else scorer
    policy = policy or (network.options.softcore if network.options is not None else SoftcorePolicy())
    _, mappings = _fixed_point(network, policy)

    edges: list[Transformation] = []
    problems: list[str] = []
    reduced = 0
    for edge in network.edges:
        mapping = mappings.get(edge.unordered_key)
        if edge.kind is EdgeKind.CBFE or mapping is None or len(mapping.cc1) == len(edge.mapping.cc1):
            edges.append(edge)
            continue
        reduced += 1
        rebuilt = _rescore(edge, network.ligands, mapping, scorer_obj, policy)
        if not rebuilt.feasible:
            reasons = ", ".join(r.value for r in rebuilt.score.rejections) or "unknown"
            problems.append(f"{rebuilt.key}: {reasons} (core {len(edge.mapping.cc1)} -> {len(mapping.cc1)} atom(s))")
        edges.append(rebuilt)

    if problems:
        raise NetworkPlanError(
            "consistency='graph' leaves these selected edge(s) infeasible:\n  "
            + "\n  ".join(problems)
            + "\nEvery ligand must keep one core across all of its edges, so the shared core is the "
            "intersection over the whole network and is necessarily smaller than any pairwise one. "
            "Loosen min_core_atoms / max_softcore_atoms, drop the ligand that pulls the intersection "
            "down, plan the subsets separately, or use consistency='pairwise'."
        )

    logger.info("consistency='graph': reduced the core on %d of %d selected edge(s)", reduced, len(network.edges))
    return replace(network, edges=tuple(edges))


def _rescore(
    edge: Transformation,
    ligands: Mapping[str, Ligand],
    mapping: AtomMapping,
    scorer: AbstractScorer,
    policy: SoftcorePolicy,
) -> Transformation:
    """Rebuild one edge around a reduced core, re-running repair, descriptors, and cost."""
    source, target = ligands[edge.source], ligands[edge.target]
    repaired, repair = repair_softcore_connectivity(source, target, mapping, policy)
    descriptors = compute_descriptors(source, target, repaired, repair)
    rejections: list[RejectionReason] = []
    if repair.rejection is not None:
        rejections.append(repair.rejection)
    elif descriptors["core_rmsd"] > policy.core_rmsd_threshold:
        rejections.append(RejectionReason.CORE_GEOMETRY_MISMATCH)
    score = scorer.score_edge(descriptors, rejections=rejections)
    return replace(edge, mapping=repaired, repair=repair, score=score)


def maybe_apply_graph_consistency(
    network: Network, options: NetworkOptions, *, scorer: AbstractScorer | str = "linear"
) -> Network:
    """Apply :func:`apply_graph_consistency` when *options* asks for it.

    A single gate, called on every path out of the pipeline, so that
    ``consistency="graph"`` cannot be honoured on one route and silently dropped on another
    -- which is the shape of the bug the option had before it did anything at all.
    """
    if options.consistency != "graph":
        return network
    return apply_graph_consistency(network, scorer=scorer, policy=options.softcore)
