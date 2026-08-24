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
from rbfenetmap.core.options import CONSISTENCY_SCOPES, NetworkOptions, SoftcorePolicy
from rbfenetmap.core.molgraph import hydrogen_parents
from rbfenetmap.core.softcore import repair_softcore_connectivity

__all__ = ("apply_graph_consistency", "consistency_groups", "graph_consistent_cores", "maybe_apply_graph_consistency")

logger = logging.getLogger(__name__)


def consistency_groups(network: Network, scope: str) -> Mapping[str, int] | None:
    """Partition the ligands into the sets that must each share one core.

    Parameters
    ----------
    network : Network
    scope : str
        A member of :data:`~rbfenetmap.core.options.CONSISTENCY_SCOPES` other than
        ``"pairwise"``, which asks for no consistency at all and never reaches here.

    Returns
    -------
    Mapping[str, int] or None
        Ligand name to group index, or ``None`` for ``"graph"`` -- one group covering
        everything, which is the same thing said without building a dictionary the callers
        would then have to check every lookup against.

    Notes
    -----
    ``"component"`` groups by the connected components of the **RBFE-only selected**
    subgraph, and each of those three words is load-bearing. Selected, because the
    candidate pool holds edges the planner rejected. RBFE-only, because a CBFE bridge joins
    components without relating any atoms, so counting it would merge two groups that share
    no scaffold and hand the intersection a pair of ligands with nothing in common. And
    components rather than clusters, because that partition is already implied by the pool:
    a set whose scaffolds cannot be mapped to each other *is* several components.
    """
    if scope == "graph":
        return None
    if scope != "component":  # pragma: no cover - guarded by NetworkOptions
        raise ValueError(f"Unknown consistency scope {scope!r}. Known: {list(CONSISTENCY_SCOPES)}.")

    import networkx as nx

    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(network.ligands)
    graph.add_edges_from(edge.unordered_key for edge in network.edges if edge.kind is EdgeKind.RBFE)
    return {
        name: index
        for index, component in enumerate(sorted(map(sorted, nx.connected_components(graph))))
        for name in component
    }


def _in_one_group(edge: Transformation, groups: Mapping[str, int] | None) -> bool:
    """Whether *edge* lies inside a single consistency group.

    A boundary edge is **exempt**, exactly as a CBFE edge is, and for the same reason: the
    two ligands are not being asked to share anything, so intersecting across the edge
    would propagate the constraint between groups and collapse the scope back to
    ``"graph"``. The price is that a ligand sitting on a boundary edge holds its
    group-uniform core on its internal edges and a different, pairwise one there.
    """
    if groups is None:
        return True
    return groups.get(edge.source) == groups.get(edge.target)


def _core_by_ligand(
    edges: Sequence[Transformation], heavy: Mapping[str, frozenset[int]], parents: Mapping[str, Mapping[int, int]]
) -> tuple[dict[str, set[int]], dict[str, dict[int, int]]]:
    """Intersect each ligand's common core over the RBFE edges incident to it.

    Returns the surviving **heavy** atoms per ligand, and how many core hydrogens each
    surviving heavy atom may keep.

    Hydrogens are intersected by *count per parent*, not by index, and that distinction is
    the whole of it. Two edges out of the same ligand routinely put **different** hydrogens
    of a symmetric group in the core -- which of a methyl's three hydrogens got paired is an
    artefact of the embedding, not chemistry. Intersecting raw indices therefore drops all
    of them while keeping their parent, and a soft-core hydrogen on a common-core parent is
    its own region, because hydrogen-follows-parent is deliberately one-way. The repair then
    has to bridge regions the intersection invented, and the cascade eats the core: measured
    on a three-scaffold set, every edge failed with the heavy core untouched at 9 atoms.

    Counting instead asks the question that has a chemical answer -- *how many* hydrogens on
    this atom are shared, not *which* -- and :func:`_restrict` then takes the lowest-indexed
    that many, which is the same choice on every edge and so is uniform by construction.
    """
    cores: dict[str, set[int]] = {}
    caps: dict[str, dict[int, int]] = {}
    for edge in edges:
        for name, atoms in ((edge.source, set(edge.mapping.cc1)), (edge.target, set(edge.mapping.cc2))):
            keep = atoms & heavy[name]
            current = cores.get(name)
            cores[name] = keep if current is None else current & keep

            counted: dict[int, int] = {}
            for atom in atoms - heavy[name]:
                parent = parents[name].get(atom)
                if parent is not None:
                    counted[parent] = counted.get(parent, 0) + 1
            seen = caps.get(name)
            if seen is None:
                caps[name] = counted
            else:
                caps[name] = {p: min(n, counted.get(p, 0)) for p, n in seen.items()}
    return cores, caps


def _restrict(
    mapping: AtomMapping,
    keep_1: set[int],
    keep_2: set[int],
    parents_1: Mapping[int, int],
    parents_2: Mapping[int, int],
    caps_1: Mapping[int, int],
    caps_2: Mapping[int, int],
) -> AtomMapping:
    """Return *mapping* with every core pair not surviving on **both** sides demoted.

    A heavy pair is kept only if each of its two atoms is in its own ligand's surviving set.
    An atom whose partner was dropped has nothing left to correspond to, so it becomes
    soft-core too -- which is exactly why one pass is not enough.

    A hydrogen pair is kept when both parents survive *and* the pair falls within both
    ligands' per-parent quota. Hydrogens are taken lowest index first, so the same ones are
    kept on every edge out of a ligand and the resulting core -- and therefore the
    soft-core -- is identical across them without ever comparing hydrogen indices between
    edges, which is the comparison that has no chemical meaning.
    """
    hydrogens = sorted(
        ((a, b) for a, b in zip(mapping.cc1, mapping.cc2) if a in parents_1 and b in parents_2),
        key=lambda pair: pair[0],
    )
    used_1: dict[int, int] = {}
    used_2: dict[int, int] = {}
    allowed: set[tuple[int, int]] = set()
    for a, b in hydrogens:
        anchor_1, anchor_2 = parents_1[a], parents_2[b]
        if anchor_1 not in keep_1 or anchor_2 not in keep_2:
            continue
        rank_1, rank_2 = used_1.get(anchor_1, 0), used_2.get(anchor_2, 0)
        if rank_1 >= caps_1.get(anchor_1, 0) or rank_2 >= caps_2.get(anchor_2, 0):
            continue
        used_1[anchor_1], used_2[anchor_2] = rank_1 + 1, rank_2 + 1
        allowed.add((a, b))

    pairs = []
    for a, b in zip(mapping.cc1, mapping.cc2):
        if a in parents_1 and b in parents_2:
            if (a, b) in allowed:
                pairs.append((a, b))
        elif a in keep_1 and b in keep_2:
            pairs.append((a, b))
    return AtomMapping.from_core_pairs(
        pairs, n_atoms_1=mapping.n_atoms_1, n_atoms_2=mapping.n_atoms_2, method=mapping.method
    )


def graph_consistent_cores(
    network: Network, *, policy: SoftcorePolicy | None = None, scope: str = "graph"
) -> dict[str, frozenset[int]]:
    """Return the atoms each ligand keeps in the graph-consistent core.

    Parameters
    ----------
    network : Network
        A planned network. Only its selected RBFE edges are read.
    policy : SoftcorePolicy, optional
        Used for the repair run between intersection passes. Defaults to the network's own
        policy, or to library defaults if the network carries no options.
    scope : str, optional
        ``"graph"`` (default) or ``"component"``. See :func:`consistency_groups`.

    Returns
    -------
    dict[str, frozenset[int]]
        Atom indices, per ligand name. A ligand with no selected RBFE edge is absent: it is
        under no constraint, because consistency is a statement about atoms shared *across*
        edges and it has none.

    Warnings
    --------
    **This is not a feasibility statement.** Unlike :func:`apply_graph_consistency`, it
    never raises, and the cores it reports can be ones no runnable network could use: when
    the repair rejects an edge it returns that edge's mapping unchanged, so a core that
    survived only because its repair failed is reported here exactly like one that survived
    on merit. Read it as "the atoms these ligands have in common", not as "the core your
    edges will run with". Call :func:`apply_graph_consistency` for the latter -- it is the
    one that checks.

    Notes
    -----
    Exposed separately from :func:`apply_graph_consistency` because the surviving core is
    the answer to "do these ligands share a scaffold at all, and how big is it?", which is
    worth asking without rewriting a network to find out.
    """
    return {
        name: frozenset(atoms)
        for name, atoms in _fixed_point(network, policy, consistency_groups(network, scope))[0].items()
    }


def _fixed_point(
    network: Network, policy: SoftcorePolicy | None, groups: Mapping[str, int] | None = None
) -> tuple[dict[str, set[int]], dict[tuple[str, str], AtomMapping]]:
    """Iterate intersect-then-repair until no core shrinks further.

    Returns the per-ligand surviving cores and the per-pair repaired mappings. Termination
    is guaranteed because every pass is non-increasing on a finite set of atoms and the loop
    exits as soon as a pass changes nothing.

    *groups* restricts which edges take part: an edge crossing two groups is left out of the
    system entirely, so it is never intersected and never repaired here. Scoping therefore
    changes *which* edges are constrained, never how the iteration converges -- the argument
    below is per ligand and per group, and is unaffected.
    """
    policy = policy or (network.options.softcore if network.options is not None else SoftcorePolicy())
    rbfe = [edge for edge in network.edges if edge.kind is EdgeKind.RBFE and _in_one_group(edge, groups)]
    mappings = {edge.unordered_key: edge for edge in rbfe}

    # Precomputed once: the heavy-atom set and the hydrogen->parent map per ligand. Both are
    # properties of the molecule, not of the iteration, and recomputing them per pass would
    # walk every RDKit mol on every pass of a loop that runs until nothing changes.
    involved = {name for edge in rbfe for name in (edge.source, edge.target)}
    heavy = {name: frozenset(network.ligands[name].heavy_indices) for name in involved}
    parents = {name: hydrogen_parents(network.ligands[name].mol) for name in involved}

    current: dict[tuple[str, str], AtomMapping] = {pair: edge.mapping for pair, edge in mappings.items()}
    # Bounded by the total core size, which strictly decreases on every pass that is not the
    # last; the +2 covers the confirming pass and the degenerate empty-network case.
    limit = sum(len(mapping.cc1) for mapping in current.values()) + 2
    cores: dict[str, set[int]] = {}
    for _ in range(limit):
        oriented = [replace(mappings[pair], mapping=current[pair]) for pair in sorted(current)]
        cores, caps = _core_by_ligand(oriented, heavy, parents)
        changed = False
        for pair, edge in sorted(mappings.items()):
            mapping = current[pair]
            keep_1 = cores.get(edge.source, set(mapping.cc1) & heavy[edge.source])
            keep_2 = cores.get(edge.target, set(mapping.cc2) & heavy[edge.target])
            restricted = _restrict(
                mapping,
                keep_1,
                keep_2,
                parents[edge.source],
                parents[edge.target],
                caps[edge.source],
                caps[edge.target],
            )
            if len(restricted.cc1) == len(mapping.cc1):
                continue
            source, target = network.ligands[edge.source], network.ligands[edge.target]
            repaired, _ = repair_softcore_connectivity(source, target, restricted, policy)
            current[pair] = repaired
            changed = True
        if not changed:
            break
    else:  # pragma: no cover - unreachable while the repair only ever shrinks a core
        # The bound above is a proof, not a check, and a proof stops holding the moment
        # someone makes the repair non-monotone. Falling out of the loop would return a
        # `cores` computed at the top of the last pass and therefore stale against
        # `current` -- a network reported as consistent that is not. Fail instead.
        raise NetworkPlanError(
            f"consistency did not converge in {limit} passes over {len(mappings)} edge(s). "
            "Every pass that changes anything must strictly shrink the total core, so this "
            "means the soft-core repair grew a core somewhere. That is a bug in the repair, "
            "not a setting you can adjust."
        )

    return cores, current


def apply_graph_consistency(
    network: Network,
    *,
    scorer: AbstractScorer | str = "linear",
    policy: SoftcorePolicy | None = None,
    scope: str = "graph",
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
    scope : str, optional
        How widely one core is required per ligand: ``"graph"`` (default) over all of its
        selected RBFE edges, or ``"component"`` only within its connected component of the
        RBFE-only selected subgraph. See
        :func:`~rbfenetmap.core.consistency.consistency_groups`.

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
    groups = consistency_groups(network, scope)
    _, mappings = _fixed_point(network, policy, groups)

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
            problems.append(_describe_failure(edge, rebuilt, network.ligands, policy))
        edges.append(rebuilt)

    if problems:
        raise NetworkPlanError(
            f"consistency={scope!r} leaves these selected edge(s) infeasible:\n  "
            + "\n  ".join(problems)
            + "\n"
            + _failure_advice(scope, policy, problems)
        )

    logger.info("consistency=%r: reduced the core on %d of %d selected edge(s)", scope, reduced, len(network.edges))
    return replace(network, edges=tuple(edges))


#: Rejections that mean "the core ended up too small", as opposed to a geometry or
#: attachment problem. These are the ones a ring-system cascade produces, so they are the
#: ones worth pointing at ``ring_policy`` for.
_COLLAPSE_REASONS = frozenset(
    {RejectionReason.CORE_TOO_SMALL, RejectionReason.SOFTCORE_TOO_LARGE, RejectionReason.SOFTCORE_FRACTION}
)


def _heavy_core(mapping: AtomMapping, ligand: Ligand) -> int:
    """Heavy atoms in *ligand*'s side of *mapping*'s common core."""
    return len(set(mapping.cc1) & set(ligand.heavy_indices))


def _describe_failure(
    original: Transformation, rebuilt: Transformation, ligands: Mapping[str, Ligand], policy: SoftcorePolicy
) -> str:
    """Explain one edge's failure using the numbers that actually caused it.

    Getting this right took two attempts, and both wrong versions are worth recording
    because the obvious fix is also wrong.

    The original reported ``len(cc1)`` before and after the intersection. That counts
    **total** atoms while ``min_core_atoms`` counts **heavy** ones, so a reader saw
    "core 15 -> 14" against a threshold of 4 and could not reconcile them.

    Reporting heavy atoms instead is necessary and not sufficient: the intersection often
    removes only *hydrogens*, leaving the heavy core untouched at, say, 9 -> 9 while the
    edge still fails. The damage is done afterwards. Dropping those atoms splits the
    soft-core into several regions, the repair bridges them by demoting more atoms, the
    closure rules cascade, and the core is eaten from the inside. None of that is visible
    in a before-and-after of the intersection, and on rejection the repair returns its
    *input* mapping, so there is no post-repair core to report either.

    So this states what consistency did, then hands over to the repair's own trace, which
    already narrates the collapse step by step and needed only to be surfaced.
    """
    reasons = ", ".join(r.value for r in rebuilt.score.rejections) or "unknown"
    source = ligands[original.source]
    before = _heavy_core(original.mapping, source)
    after = _heavy_core(rebuilt.mapping, source)
    head = f"{rebuilt.key}: {reasons}"
    if before != after:
        head += f" (heavy core {before} -> {after} on intersection"
    else:
        head += f" (intersection left the heavy core at {before}, dropping hydrogens only"
    if RejectionReason.CORE_TOO_SMALL in rebuilt.score.rejections:
        head += f"; min_core_atoms={policy.min_core_atoms}"
    head += ")"

    # The repair's trace is the part that explains it: how many regions the intersection
    # left, what bridging them cost, and what the closure then pulled in.
    detail = [line for line in rebuilt.repair.trace if "initial:" in line or "iter" in line]
    if detail:
        head += "".join(f"\n      {line}" for line in detail[-3:])
    return head


def _failure_advice(scope: str, policy: SoftcorePolicy, problems: Sequence[str]) -> str:
    """The paragraph after the per-edge lines: why this happened and what to change.

    Names ``ring_policy`` first when the cores collapsed under ``ring_system``, because that
    is usually the actual cause and loosening ``min_core_atoms`` will not help: intersecting
    removes one ring atom, whole-ring closure then demotes the entire ring, and the core is
    gone well before any count is consulted. Recommending the threshold in that case sends
    the reader to a knob that cannot fix it -- the previous version of this message did.
    """
    lines = [
        "Every ligand must keep one core across all of the edges in its consistency group, so "
        "the shared core is an intersection and is necessarily no larger than any pairwise one."
    ]
    collapsed = any(reason.value in problem for problem in problems for reason in _COLLAPSE_REASONS)
    if collapsed:
        lines.append(
            "The traces above are the thing to read. These cores were not shrunk to death by the "
            "intersection itself -- they were eaten by the repair that followed it, because "
            "dropping the non-shared atoms split the soft-core into several regions and bridging "
            "those cost more atoms than the intersection ever did."
        )
        if policy.ring_policy == "ring_system":
            lines.append(
                "With ring_policy='ring_system' that bridging cascades: touching one ring atom "
                "demotes the whole ring. Loosening min_core_atoms does not help against that, and "
                "ring_policy='none' is the knob that speaks to it."
            )
    if scope == "graph":
        lines.append(
            "consistency='component' asks for the same rule within each connected component of "
            "the RBFE network instead of across all of it, which is what a set of several "
            "scaffolds usually wants."
        )
    lines.append(
        "Otherwise: drop the ligand that pulls the intersection down, plan the subsets "
        "separately, or use consistency='pairwise'."
    )
    return " ".join(lines)


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

    A single gate, called on every path out of the pipeline, so that a consistency scope
    cannot be honoured on one route and silently dropped on another -- which is the shape of
    the bug the option had before it did anything at all.

    ``"pairwise"`` is the only scope that does nothing, and it is tested for by name rather
    than by position in the ladder: a scope added later should have to state that it is a
    no-op, not inherit it from being listed first.
    """
    if options.consistency == "pairwise":
        return network
    return apply_graph_consistency(network, scorer=scorer, policy=options.softcore, scope=options.consistency)
