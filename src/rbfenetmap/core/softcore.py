"""Soft-core connectivity repair.

This module enforces the constraints the whole package is organised around: **a
transformation has at most one connected soft-core region per side, and each region
attaches to the common core through exactly one bond**. A mapper is free
to return a correspondence whose unmapped atoms fall into several disconnected pieces --
that is the normal outcome for, say, a benzene to *para*-xylene transformation, where
two hydrogens on opposite sides of the ring both disappear. Such a partition cannot be
run as a single alchemical transformation, so it must either be repaired or rejected.

The repair works by *demoting* common-core atoms into the soft-core until the pieces
join up. Choosing which atoms to demote is a node-weighted Steiner tree problem: the
soft-core fragments are the terminals, the bond graph is the network, and the cost of
recruiting an atom is how much soft-core that recruitment ultimately drags in.

Three closure rules run to a fixpoint after every recruitment:

**(a) whole-ring** -- a ring is never left half soft-core. Touching one ring atom absorbs
the ring. Fused systems then cascade on their own, because absorbing one ring pulls in
the atoms it shares with its neighbours, which makes those neighbours
intersected-but-not-contained on the next sweep.

**(b) hydrogen-follows-parent** -- a hydrogen joins the soft-core when its heavy parent
does. Deliberately one-way: a soft-core hydrogen whose parent is common core stays put
as a one-atom region. That asymmetry is not an oversight. ``R-H -> R-CH3`` is the single
most common transformation in the field, and its soft-core on the ``R-H`` side is exactly
one hydrogen attached to a core carbon. A two-way rule would demote that carbon, then its
ring, and destroy the edge.

**(c) mapped-partner** -- demoting an atom demotes whatever it is mapped to. This is the
only rule that couples the two molecules, and it is why the repair is genuinely joint:
fixing a fragmentation on side 1 can create a new one on side 2, which the loop must then
fix in turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import networkx as nx

from rbfenetmap.core.exceptions import RepairError
from rbfenetmap.core.models import AtomMapping, Ligand, RejectionReason, SoftcoreRepair
from rbfenetmap.core.molgraph import (
    connected_components_of,
    hydrogen_parents,
    mol_to_graph,
    node_weighted_steiner,
    ring_systems,
)
from rbfenetmap.core.options import SoftcorePolicy

__all__ = (
    "RepairContext",
    "detect_fragments",
    "joint_closure",
    "precheck_mapping",
    "repair_softcore_connectivity",
    "softcore_attachment_edges",
)


def detect_fragments(graph: nx.Graph, softcore: Iterable[int]) -> list[set[int]]:
    """Return the connected components of the soft-core, largest first.

    An empty soft-core yields an empty list. Zero regions is legal -- the constraint is
    *at most* one region, and a transformation that only reorders a common core has none.
    """
    return connected_components_of(graph, softcore)


def softcore_attachment_edges(graph: nx.Graph, softcore: Iterable[int]) -> list[tuple[int, int]]:
    """Return bonds crossing from the soft-core to the common core.

    Each tuple is oriented ``(softcore_atom, common_core_atom)``. Counting edges, rather
    than distinct common-core atoms, expresses the alchemical topology rule directly:
    one soft-core region must be a singly attached substituent, not a bridge or ring path.
    """
    softcore_set = set(softcore)
    return sorted(
        (atom, neighbor)
        for atom in softcore_set
        for neighbor in graph.neighbors(atom)
        if neighbor not in softcore_set
    )


def _attachment_check(
    softcore_1: set[int], softcore_2: set[int], context: RepairContext, trace: list[str]
) -> RejectionReason | None:
    """Reject a soft-core region connected to the common core by multiple bonds."""
    for side, softcore in ((1, softcore_1), (2, softcore_2)):
        if not softcore:
            continue
        graph, _, _, _ = context.side(side)
        attachments = softcore_attachment_edges(graph, softcore)
        if len(attachments) != 1:
            trace.append(
                f"final side {side}: rejected ({RejectionReason.SOFTCORE_MULTIPLE_ATTACHMENTS.value}); "
                f"soft-core has {len(attachments)} common-core attachment bond(s) {attachments}"
            )
            return RejectionReason.SOFTCORE_MULTIPLE_ATTACHMENTS
    return None


@dataclass
class RepairContext:
    """Everything the repair needs about one candidate pair, precomputed.

    Built once per edge by :meth:`build`. Holding the bond graphs, ring lists, hydrogen
    parentage, and demotion costs here keeps them out of the repair loop, which would
    otherwise recompute them on every iteration.
    """

    graph_1: nx.Graph
    graph_2: nx.Graph
    rings_1: tuple[frozenset[int], ...]
    rings_2: tuple[frozenset[int], ...]
    hydrogen_parent_1: dict[int, int]
    hydrogen_parent_2: dict[int, int]
    forward: dict[int, int]
    reverse: dict[int, int]
    heavy_1: frozenset[int]
    heavy_2: frozenset[int]
    policy: SoftcorePolicy
    n_atoms_1: int = 0
    n_atoms_2: int = 0
    _cost_cache: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, source: Ligand, target: Ligand, mapping: AtomMapping, policy: SoftcorePolicy) -> "RepairContext":
        """Precompute the graphs, rings, hydrogen parentage, and correspondence.

        Raises
        ------
        RepairError
            If the mapping's atom counts disagree with the ligands. This is a programming
            error rather than a chemistry one -- a mapping built for a different pair.
        """
        if mapping.n_atoms_1 != source.n_atoms or mapping.n_atoms_2 != target.n_atoms:
            raise RepairError(
                f"Mapping describes molecules of {mapping.n_atoms_1}/{mapping.n_atoms_2} atoms but "
                f"{source.name}/{target.name} have {source.n_atoms}/{target.n_atoms}."
            )
        return cls(
            graph_1=mol_to_graph(source.mol),
            graph_2=mol_to_graph(target.mol),
            rings_1=ring_systems(source.mol),
            rings_2=ring_systems(target.mol),
            hydrogen_parent_1=hydrogen_parents(source.mol),
            hydrogen_parent_2=hydrogen_parents(target.mol),
            forward=mapping.forward,
            reverse=mapping.reverse,
            heavy_1=frozenset(source.heavy_indices),
            heavy_2=frozenset(target.heavy_indices),
            policy=policy,
            n_atoms_1=source.n_atoms,
            n_atoms_2=target.n_atoms,
        )

    def side(self, side: int) -> tuple[nx.Graph, tuple[frozenset[int], ...], dict[int, int], frozenset[int]]:
        """Return ``(graph, rings, hydrogen_parents, heavy_indices)`` for *side*."""
        if side == 1:
            return self.graph_1, self.rings_1, self.hydrogen_parent_1, self.heavy_1
        return self.graph_2, self.rings_2, self.hydrogen_parent_2, self.heavy_2

    def n_heavy(self, side: int) -> int:
        """Heavy-atom count for *side*."""
        return len(self.heavy_1 if side == 1 else self.heavy_2)

    def demote_cost(self, atom: int, side: int) -> float:
        """Cost of demoting *atom*, measured as the closure it triggers.

        Parameters
        ----------
        atom : int
            The common-core atom under consideration.
        side : int
            ``1`` or ``2``.

        Returns
        -------
        float
            Total atoms across *both* molecules that demoting this one atom ultimately
            pulls into the soft-core.

        Notes
        -----
        Using the closure size as the cost is what makes the Steiner search behave
        chemically without any hand-tuned table of per-element weights. A hydrogen costs
        about 2 (itself and its partner). A peripheral heavy atom costs a little more. An
        aromatic carbon costs its entire fused ring system, plus every attached hydrogen,
        plus all of their partners on the other side -- so the solver routes around rings
        whenever an acyclic path exists, and only pays for a ring when there is no
        alternative.

        The cost is measured in isolation, from an empty soft-core, so it slightly
        overestimates once part of the closure is already soft-core. It is a search
        heuristic, not an accounting of the final result, and computing it once keeps the
        repair loop cheap.
        """
        key = (side, atom)
        cached = self._cost_cache.get(key)
        if cached is not None:
            return cached
        seed_1 = {atom} if side == 1 else set()
        seed_2 = {atom} if side == 2 else set()
        closed_1, closed_2 = joint_closure(seed_1, seed_2, self)
        cost = float(len(closed_1) + len(closed_2))
        self._cost_cache[key] = cost
        return cost

    def cost_fn(self, side: int) -> Callable[[int], float]:
        """Return a one-argument cost callable bound to *side*."""

        def _cost(atom: int) -> float:
            return self.demote_cost(atom, side)

        return _cost


def joint_closure(softcore_1: set[int], softcore_2: set[int], context: RepairContext) -> tuple[set[int], set[int]]:
    """Apply the three closure rules to a fixpoint over both sides.

    Parameters
    ----------
    softcore_1, softcore_2 : set[int]
        Current soft-core atom sets. Not modified in place.
    context : RepairContext
        Precomputed graphs, rings, hydrogen parentage, and correspondence.

    Returns
    -------
    tuple[set[int], set[int]]
        The closed soft-core sets.

    Notes
    -----
    Terminates because every rule only ever adds atoms, and the atom sets are finite.
    """
    side_1 = set(softcore_1)
    side_2 = set(softcore_2)
    apply_rings = context.policy.ring_policy == "ring_system"

    changed = True
    while changed:
        changed = False

        for current, (_, rings, parents, _) in ((side_1, context.side(1)), (side_2, context.side(2))):
            # (a) whole-ring: never leave a ring half soft-core.
            if apply_rings:
                for ring in rings:
                    if current & ring and not ring <= current:
                        current |= ring
                        changed = True
            # (b) hydrogen-follows-parent, one-way only. See the module docstring.
            for hydrogen, parent in parents.items():
                if parent in current and hydrogen not in current:
                    current.add(hydrogen)
                    changed = True

        # (c) mapped-partner: the only rule coupling the two molecules.
        new_2 = {context.forward[a] for a in side_1 if a in context.forward} - side_2
        if new_2:
            side_2 |= new_2
            changed = True
        new_1 = {context.reverse[b] for b in side_2 if b in context.reverse} - side_1
        if new_1:
            side_1 |= new_1
            changed = True

    return side_1, side_2


def _heavy_count(atoms: Iterable[int], heavy: frozenset[int]) -> int:
    """Number of non-hydrogen atoms in *atoms*."""
    return sum(1 for a in atoms if a in heavy)


def _budget_check(softcore_1: set[int], softcore_2: set[int], context: RepairContext) -> RejectionReason | None:
    """Return why the current soft-core sets are unacceptable, or ``None``.

    Checks run in order of severity, so the reported reason is the most fundamental
    problem rather than whichever threshold happened to be tightest.
    """
    policy = context.policy

    if len(softcore_1) >= context.n_atoms_1 or len(softcore_2) >= context.n_atoms_2:
        return RejectionReason.NO_COMMON_CORE

    core_heavy = sum(1 for a in context.forward if a not in softcore_1 and a in context.heavy_1)
    if core_heavy < policy.min_core_atoms:
        return RejectionReason.CORE_TOO_SMALL

    heavy_1 = _heavy_count(softcore_1, context.heavy_1)
    heavy_2 = _heavy_count(softcore_2, context.heavy_2)
    if max(heavy_1, heavy_2) > policy.max_softcore_atoms:
        return RejectionReason.SOFTCORE_TOO_LARGE

    for heavy, side in ((heavy_1, 1), (heavy_2, 2)):
        total = context.n_heavy(side)
        if total and heavy / total > policy.max_softcore_fraction:
            return RejectionReason.SOFTCORE_FRACTION

    return None


def precheck_mapping(
    source: Ligand, target: Ligand, mapping: AtomMapping, policy: SoftcorePolicy
) -> RejectionReason | None:
    """Cheap rejections applied before the repair runs.

    Parameters
    ----------
    source, target : Ligand
        The two ligands.
    mapping : AtomMapping
        The mapper's raw output.
    policy : SoftcorePolicy
        Feasibility thresholds.

    Returns
    -------
    RejectionReason or None

    Notes
    -----
    Ordering matters for cost, not just for message quality. A scaffold hop whose common
    core covers a tenth of either molecule will certainly fail the soft-core budget, but
    only after the Steiner solver has done real work on a large fragmented soft-core.
    Catching it on the MCS fraction first skips that entirely.
    """
    heavy_core = sum(1 for a in mapping.cc1 if a in set(source.heavy_indices))
    if heavy_core == 0:
        return RejectionReason.NO_COMMON_CORE

    smaller = min(source.n_heavy, target.n_heavy)
    if smaller and heavy_core / smaller < policy.min_mcs_fraction:
        return RejectionReason.MCS_FRACTION_TOO_LOW

    if policy.charge_change_policy == "reject" and source.charge != target.charge:
        return RejectionReason.NET_CHARGE_CHANGE

    return None


def repair_softcore_connectivity(
    source: Ligand, target: Ligand, mapping: AtomMapping, policy: SoftcorePolicy | None = None
) -> tuple[AtomMapping, SoftcoreRepair]:
    """Repair the mapping so each side has at most one connected soft-core region.

    Parameters
    ----------
    source, target : Ligand
        The two ligands.
    mapping : AtomMapping
        The mapper's output, whose soft-core may be fragmented.
    policy : SoftcorePolicy, optional
        Feasibility thresholds and the ring policy. Defaults are used if omitted.

    Returns
    -------
    tuple[AtomMapping, SoftcoreRepair]
        The repaired mapping and a record of what was done. On rejection the *original*
        mapping is returned unchanged alongside a repair carrying the
        :class:`~rbfenetmap.core.models.RejectionReason`: an edge that will not be used
        should not be silently mutated, and the caller may still want to show the user
        the mapping that failed.

    Raises
    ------
    rbfenetmap.core.exceptions.RepairError
        Only for malformed input -- a mapping that does not describe these molecules, or
        a molecule whose bond graph is disconnected. An edge that genuinely cannot be
        repaired is *not* an error; it comes back as a rejection.

    Notes
    -----
    The loop terminates in at most ``n_atoms_1 + n_atoms_2`` iterations. Both soft-core
    sets grow monotonically within finite atom sets, and any iteration that does not
    return adds at least one atom, because a bridge joining two or more fragments must
    contain an atom that is not already soft-core.
    """
    policy = policy or SoftcorePolicy()
    context = RepairContext.build(source, target, mapping, policy)
    trace: list[str] = []

    # Normalise the mapper's raw output first. Mappers are not required to respect the
    # ring or hydrogen rules, so the sets they hand over are frequently not yet closed.
    softcore_1, softcore_2 = joint_closure(set(mapping.sc1), set(mapping.sc2), context)
    if len(softcore_1) != len(mapping.sc1) or len(softcore_2) != len(mapping.sc2):
        trace.append(
            f"closure: normalised raw soft-core {len(mapping.sc1)}/{len(mapping.sc2)} -> "
            f"{len(softcore_1)}/{len(softcore_2)} atoms"
        )

    fragments_1 = detect_fragments(context.graph_1, softcore_1)
    fragments_2 = detect_fragments(context.graph_2, softcore_2)
    before = (len(fragments_1), len(fragments_2))
    trace.append(f"initial: {before[0]} soft-core region(s) on side 1, {before[1]} on side 2")

    max_iterations = policy.max_iterations or (source.n_atoms + target.n_atoms)
    iterations = 0
    rejection: RejectionReason | None = None

    for iterations in range(1, max_iterations + 1):
        fragments_1 = detect_fragments(context.graph_1, softcore_1)
        fragments_2 = detect_fragments(context.graph_2, softcore_2)
        if len(fragments_1) <= 1 and len(fragments_2) <= 1:
            iterations -= 1  # the final pass only confirmed the result; it changed nothing
            break

        for side, fragments, current in ((1, fragments_1, softcore_1), (2, fragments_2, softcore_2)):
            if len(fragments) <= 1:
                continue
            graph, _, _, _ = context.side(side)
            try:
                recruited, approximate = node_weighted_steiner(graph, fragments, context.cost_fn(side))
            except ValueError as exc:
                raise RepairError(
                    f"{source.name}~{target.name}: cannot bridge the soft-core on side {side}: {exc}"
                ) from exc
            current |= recruited
            trace.append(
                f"iter {iterations} side {side}: bridged {len(fragments)} regions by demoting "
                f"{len(recruited)} atom(s) {sorted(recruited)}" + (" [steiner:approximate]" if approximate else "")
            )

        closed_1, closed_2 = joint_closure(softcore_1, softcore_2, context)
        grown = (len(closed_1) - len(softcore_1), len(closed_2) - len(softcore_2))
        softcore_1, softcore_2 = closed_1, closed_2
        if any(grown):
            trace.append(f"iter {iterations} closure: +{grown[0]} atom(s) on side 1, +{grown[1]} on side 2")

        rejection = _budget_check(softcore_1, softcore_2, context)
        if rejection is not None:
            trace.append(f"iter {iterations}: rejected ({rejection.value})")
            break
    else:
        rejection = RejectionReason.REPAIR_DID_NOT_CONVERGE
        trace.append(f"exhausted {max_iterations} iterations without converging")

    # A repair that never had to bridge anything still has to pass the budget: a mapping
    # can arrive already connected but far too large to be worth running.
    if rejection is None:
        rejection = _budget_check(softcore_1, softcore_2, context)
        if rejection is not None:
            trace.append(f"final: rejected ({rejection.value})")

    if rejection is None:
        rejection = _attachment_check(softcore_1, softcore_2, context, trace)

    fragments_1 = detect_fragments(context.graph_1, softcore_1)
    fragments_2 = detect_fragments(context.graph_2, softcore_2)
    after = (len(fragments_1), len(fragments_2))

    demoted_1 = tuple(sorted(softcore_1 - set(mapping.sc1)))
    demoted_2 = tuple(sorted(softcore_2 - set(mapping.sc2)))

    record = SoftcoreRepair(
        applied=bool(demoted_1 or demoted_2),
        n_fragments_before=before,
        n_fragments_after=after,
        demoted_1=demoted_1,
        demoted_2=demoted_2,
        iterations=iterations,
        rejection=rejection,
        trace=tuple(trace),
    )

    if rejection is not None:
        return mapping, record

    repaired = AtomMapping.from_core_pairs(
        {a: b for a, b in context.forward.items() if a not in softcore_1 and b not in softcore_2},
        n_atoms_1=mapping.n_atoms_1,
        n_atoms_2=mapping.n_atoms_2,
        method=mapping.method,
    )
    if set(repaired.sc1) != softcore_1 or set(repaired.sc2) != softcore_2:  # pragma: no cover - defensive
        raise RepairError(
            f"{source.name}~{target.name}: repaired mapping disagrees with the closed soft-core sets. "
            "This means the partner rule did not reach a fixpoint, which should be impossible."
        )
    trace.append(f"final: {after[0]}/{after[1]} region(s), soft-core {len(softcore_1)}/{len(softcore_2)} atom(s)")
    return repaired, SoftcoreRepair(
        applied=record.applied,
        n_fragments_before=before,
        n_fragments_after=after,
        demoted_1=demoted_1,
        demoted_2=demoted_2,
        iterations=iterations,
        rejection=None,
        trace=tuple(trace),
    )
