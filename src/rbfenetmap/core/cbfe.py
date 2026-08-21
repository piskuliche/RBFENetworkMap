"""Counterpoised binding free energy (CBFE) edges.

A CBFE edge is two absolute calculations run simultaneously in opposite directions: one
ligand decoupling from the site as the other couples into it. It yields the same relative
quantity an RBFE edge does, by a different route, and the difference that matters to a
network planner is that **neither molecule is morphed into the other**. There is no common
core to find, no soft-core region to repair, and therefore no way for a CBFE edge to be
infeasible. It exists between every pair of ligands.

That makes it the natural repair for the failure mode this package hits on real ligand
sets: an MCS-based candidate pool that comes back in several disconnected pieces, with no
RBFE edge able to cross between them. CBFE edges can cross, and the cost model here is
what keeps them from being used anywhere else.

Cost
----
``cbfe_base_cost + cbfe_atom_weight * (n_heavy_1 + n_heavy_2)``, deliberately on the same
scale as the scorer's edge totals so the two kinds can be compared by the planner. The
default base, 8.0, is the linear scorer's charge-change ceiling: a CBFE edge is priced at
roughly the most expensive thing that can happen to a still-feasible RBFE edge. Realistic
totals land in ~[9, 13] against ~0.3 for a good RBFE edge and ~5-6 for a bad one, so CBFE
never wins on price -- only on being available where nothing else is. The per-atom term
exists because a counterpoised calculation decouples both molecules in full, so its
expense really does scale with how much there is to decouple.

Bridge selection
----------------
Choosing *which* CBFE edges join the subnetworks is a separate question from cost, and is
answered by :func:`select_cbfe_bridges`: a maximum-merit spanning forest over the
component quotient graph, where merit combines pairwise similarity with how well connected
each endpoint is inside its own subnetwork. See :func:`bridge_rank_key` for why both terms
are there.

The forest sweep itself is :func:`select_bridges`, which takes the partition as an argument
instead of deriving it. Connected components are only one interesting partition of a ligand
set -- a chemical clustering is another, and joining *those* groups is the same problem with
the same ranking. Keeping the partition a parameter is what lets clustered planning in
:mod:`rbfenetmap.core.clustering` reuse this machinery rather than grow a second copy of it
that would drift.
"""

from __future__ import annotations

from itertools import combinations
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from rbfenetmap.core.models import AtomMapping, EdgeKind, EdgeScore, Ligand, SoftcoreRepair, Transformation

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx

    from rbfenetmap.core.options import NetworkOptions

__all__ = (
    "BRIDGE_CENTRALITY_WEIGHT",
    "CBFE_METHOD",
    "CBFE_SCORER",
    "bridge_rank_key",
    "build_cbfe_pool",
    "cbfe_cost",
    "component_centrality",
    "make_cbfe_transformation",
    "select_bridges",
    "select_cbfe_bridges",
)

#: Recorded as :attr:`~rbfenetmap.core.models.AtomMapping.method` on a CBFE edge. It names
#: the reason the mapping is empty rather than leaving it as ``"unknown"``, which would be
#: indistinguishable from a mapper that failed.
CBFE_METHOD = "cbfe"

#: Recorded as :attr:`~rbfenetmap.core.models.EdgeScore.scorer`. No scoring plugin is
#: involved: the cost is closed-form, so attributing it to the configured scorer would be
#: a lie that shows up in the exported JSON.
CBFE_SCORER = "cbfe"

#: How much endpoint centrality is worth relative to similarity when ranking bridges.
#: Both inputs are on [0, 1], so this reads as "being maximally well connected inside both
#: subnetworks is worth as much as a 0.5 jump in Tanimoto similarity". Kept a module
#: constant rather than a user option: it is a tie-break heuristic, and the option surface
#: is already three knobs wide. It is the obvious thing to promote if it ever needs tuning.
BRIDGE_CENTRALITY_WEIGHT = 0.5


def cbfe_cost(source: Ligand, target: Ligand, options: "NetworkOptions") -> float:
    """Return the cost of the CBFE edge between two ligands.

    Parameters
    ----------
    source, target : Ligand
    options : NetworkOptions
        Supplies ``cbfe_base_cost`` and ``cbfe_atom_weight``.

    Returns
    -------
    float
        Always finite -- a CBFE edge cannot be infeasible.
    """
    return options.cbfe_base_cost + options.cbfe_atom_weight * (source.n_heavy + target.n_heavy)


def make_cbfe_transformation(source: Ligand, target: Ligand, options: "NetworkOptions") -> Transformation:
    """Build the CBFE edge between two ligands.

    The mapping is the empty-core partition: nothing in common, every atom on both sides
    soft-core. That is not a placeholder standing in for a mapping nobody computed -- it is
    a literal description of the experiment, in which both molecules are decoupled in
    full. It also satisfies every :class:`~rbfenetmap.core.models.AtomMapping` invariant,
    so the edge is a first-class transformation rather than a special case downstream.

    Parameters
    ----------
    source, target : Ligand
    options : NetworkOptions

    Returns
    -------
    Transformation
        Always feasible, with :attr:`~rbfenetmap.core.models.EdgeKind.CBFE` as its kind.
    """
    cost = cbfe_cost(source, target, options)
    mapping = AtomMapping(
        cc1=(),
        cc2=(),
        sc1=tuple(range(source.n_atoms)),
        sc2=tuple(range(target.n_atoms)),
        n_atoms_1=source.n_atoms,
        n_atoms_2=target.n_atoms,
        method=CBFE_METHOD,
    )
    score = EdgeScore(
        total=cost,
        feasible=True,
        descriptors=MappingProxyType({"n_heavy_1": float(source.n_heavy), "n_heavy_2": float(target.n_heavy)}),
        contributions=MappingProxyType(
            {
                "cbfe_base": options.cbfe_base_cost,
                "cbfe_size": options.cbfe_atom_weight * (source.n_heavy + target.n_heavy),
            }
        ),
        scorer=CBFE_SCORER,
    )
    return Transformation(
        source=source.name,
        target=target.name,
        mapping=mapping,
        repair=SoftcoreRepair(),
        score=score,
        kind=EdgeKind.CBFE,
    )


def build_cbfe_pool(
    ligands: Mapping[str, Ligand], options: "NetworkOptions", *, exclude: Iterable[tuple[str, str]] = ()
) -> dict[tuple[str, str], float]:
    """Return every eligible CBFE pair mapped to its cost.

    Costs, not :class:`~rbfenetmap.core.models.Transformation` objects. The pool is
    quadratic in the ligand count and only a handful of its entries are ever selected, so
    materializing the rest would build tens of thousands of objects -- each carrying full
    soft-core index tuples over every atom -- to throw them away. The planner builds the
    transformation for a pair once it has decided to keep it.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
    options : NetworkOptions
        ``banned_edges`` is honoured here; a ban applies to CBFE exactly as it does to
        RBFE, since it expresses "do not run this pair" rather than "do not map it".
    exclude : Iterable[tuple[str, str]], optional
        Unordered pairs to omit -- in practice the pairs that already have a feasible RBFE
        candidate. Filtering them here, once, is what keeps a pair from being offered as
        both kinds and tripping :meth:`~rbfenetmap.core.models.Network.validate`'s
        duplicate check much later.

    Returns
    -------
    dict[tuple[str, str], float]
        Keyed by sorted endpoint pair.

    Notes
    -----
    The pool spans *all* unordered pairs and deliberately ignores both ``pair_strategy``
    and the fingerprint prefilter. Those exist to hold down the number of MCS searches, and
    a CBFE edge runs none. Applying the prefilter here would be worse than merely
    unnecessary: it drops dissimilar pairs, and dissimilar pairs are precisely the ones
    that need a bridge, because their similar neighbours already had RBFE edges available.
    """
    banned = options.banned_pairs
    skip = set(exclude) | banned
    pool: dict[tuple[str, str], float] = {}
    for name_1, name_2 in combinations(sorted(ligands), 2):
        pair = (name_1, name_2)
        if pair in skip:
            continue
        pool[pair] = cbfe_cost(ligands[name_1], ligands[name_2], options)
    return pool


def component_centrality(graph: "nx.Graph", components: Sequence[set[str]]) -> dict[str, float]:
    """Score each ligand by how well connected it is *within its own* subnetwork.

    Degree divided by the largest degree in the same component, so the result is on [0, 1]
    and comparable across components of very different sizes.

    Max-normalising rather than using :func:`networkx.degree_centrality` (degree over
    ``size - 1``) matters for the singleton case. A ligand nothing could be mapped to forms
    a component of one with degree 0, and ``degree_centrality`` would score it 0.0 -- the
    worst possible entry point. But a singleton has exactly *one* way into the network, and
    ranking its only option last is backwards. Here it scores 1.0.

    Parameters
    ----------
    graph : networkx.Graph
        The feasible RBFE graph, before any bridges are added.
    components : Sequence[set[str]]
        Its connected components.

    Returns
    -------
    dict[str, float]
    """
    centrality: dict[str, float] = {}
    for component in components:
        peak = max((graph.degree(node) for node in component), default=0)
        for node in component:
            centrality[node] = 1.0 if peak == 0 else graph.degree(node) / peak
    return centrality


def bridge_rank_key(
    pair: tuple[str, str],
    *,
    similarity: Mapping[tuple[str, str], float],
    centrality: Mapping[str, float],
    cost: Mapping[tuple[str, str], float],
) -> tuple[float, float, tuple[str, str]]:
    """Rank one candidate bridge. Lower sorts better.

    Merit is a *sum* of two terms both on [0, 1]::

        merit = similarity + BRIDGE_CENTRALITY_WEIGHT * mean(centrality of the endpoints)

    Similarity is there because a bridge between chemically close ligands is the one most
    likely to give a trustworthy number even though it is being run as two absolute
    calculations. Centrality is there because a bridge landing on a hub propagates through
    the subnetwork -- it participates in cycles and shares its endpoints with many RBFE
    edges -- whereas one landing on a leaf leaves a dangling path that nothing checks.

    A sum rather than a product so the trade-off stays legible: the weight states exactly
    what one term is worth in units of the other, which a product cannot do.

    Parameters
    ----------
    pair : tuple[str, str]
        Sorted endpoint pair.
    similarity : Mapping[tuple[str, str], float]
        Fingerprint similarity per pair.
    centrality : Mapping[str, float]
        Per-ligand centrality from :func:`component_centrality`.
    cost : Mapping[tuple[str, str], float]
        CBFE cost per pair, used only to break merit ties toward the cheaper edge.

    Returns
    -------
    tuple[float, float, tuple[str, str]]
        Ends in the pair itself, so the ordering is total and the selection is
        reproducible regardless of dictionary iteration order.
    """
    merit = similarity.get(pair, 0.0) + BRIDGE_CENTRALITY_WEIGHT * 0.5 * (
        centrality.get(pair[0], 0.0) + centrality.get(pair[1], 0.0)
    )
    return (-merit, cost.get(pair, 0.0), pair)


def select_bridges(
    partition: Mapping[str, int],
    ligands: Mapping[str, Ligand],
    pool: Mapping[tuple[str, str], float],
    *,
    graph: "nx.Graph | None" = None,
    n_per_pair: int = 1,
) -> list[tuple[str, str]]:
    """Choose the edges that join the groups of *partition* into one network.

    A maximum-merit spanning forest over the *quotient graph of the partition*, by the same
    union-find sweep the planner's Kruskal pass uses. With ``g`` groups exactly ``g - 1``
    group pairs must be joined, and picking which ``g - 1`` is precisely a spanning
    selection -- ranking each group *pair* separately would produce ``C(g, 2)`` winners and
    still leave the same problem to solve.

    The partition is a **parameter** rather than something derived here, and that is what
    makes the function reusable. :func:`select_cbfe_bridges` passes the connected components
    of the feasible RBFE graph, which is the "these pieces cannot reach each other" case.
    Clustered planning passes the clusterer's partition, which is the "these pieces *can*
    reach each other but the budget is better spent inside them" case. Both want the same
    thing -- the most trustworthy few edges crossing a boundary -- and neither wants to
    reimplement the ranking.

    Parameters
    ----------
    partition : Mapping[str, int]
        Group index per ligand name. Pairs whose endpoints are missing from it are ignored.
    ligands : Mapping[str, Ligand]
    pool : Mapping[tuple[str, str], float]
        Eligible pairs and their costs. CBFE costs from :func:`build_cbfe_pool` for the
        connectivity case, RBFE edge costs for the clustered case.
    graph : networkx.Graph, optional
        The graph the endpoints live in, used only for :func:`component_centrality`. Not
        mutated. Without it, centrality contributes nothing and bridges rank on similarity
        and cost alone -- which is the right degradation, since "how well connected is this
        ligand" has no answer without a graph to ask it of.
    n_per_pair : int, optional
        Edges to take across each joined group pair. ``1`` gives the minimal spanning
        selection. ``2`` puts the crossing itself on a cycle: two edges between the same two
        groups, plus the paths inside each group, form a loop through both of them -- which
        applies the every-edge-in-a-cycle invariant precisely to the edges that most need
        checking.

    Returns
    -------
    list[tuple[str, str]]
        Sorted endpoint pairs, in selection order. Shorter than ``n_per_pair * (g - 1)``
        whenever the pool cannot supply that many crossings -- every remaining pair having
        been banned, or simply absent. The caller reports that; this function does not
        raise, because the planner has a much better diagnostic to hand than anything
        available here.

    Notes
    -----
    Centrality is computed once, on the graph as given, and not refreshed as bridges are
    added. Refreshing would make the outcome depend on selection order and would also
    contradict the intent: the question is how well connected a ligand is inside the group
    it came from, not how connected it became by being chosen.
    """
    if not pool or n_per_pair < 1:
        return []

    groups: dict[int, set[str]] = {}
    for name, index in partition.items():
        groups.setdefault(index, set()).add(name)
    if len(groups) < 2:
        return []

    crossing = [
        pair
        for pair in pool
        if pair[0] in partition and pair[1] in partition and partition[pair[0]] != partition[pair[1]]
    ]
    if not crossing:
        return []

    from rbfenetmap.core.pairs import fingerprint_pair_similarities

    similarity = fingerprint_pair_similarities(ligands, crossing)
    centrality = component_centrality(graph, [groups[index] for index in sorted(groups)]) if graph is not None else {}
    ranked = sorted(crossing, key=lambda p: bridge_rank_key(p, similarity=similarity, centrality=centrality, cost=pool))

    parent = {index: index for index in groups}

    def find(index: int) -> int:
        """Union-find with path compression, over group indices."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    bridges: list[tuple[str, str]] = []
    taken: dict[tuple[int, int], int] = {}
    needed = len(groups) - 1
    for pair in ranked:
        first, second = partition[pair[0]], partition[pair[1]]
        key = (min(first, second), max(first, second))
        if key in taken:
            # An already-joined group pair may still be owed its remaining crossings. The
            # union-find below would reject them as redundant, which is exactly right for
            # connectivity and exactly wrong for putting the crossing on a cycle.
            if taken[key] < n_per_pair:
                taken[key] += 1
                bridges.append(pair)
        elif find(first) != find(second):
            parent[find(second)] = find(first)
            taken[key] = 1
            bridges.append(pair)
        if len(taken) == needed and all(count >= n_per_pair for count in taken.values()):
            break
    return bridges


def select_cbfe_bridges(
    graph: "nx.Graph", ligands: Mapping[str, Ligand], pool: Mapping[tuple[str, str], float]
) -> list[tuple[str, str]]:
    """Choose the CBFE edges that join *graph* into one component.

    A thin caller of :func:`select_bridges` over the partition induced by *graph*'s own
    connected components: with ``c`` components, exactly ``c - 1`` bridges. One crossing per
    joined pair, because a second CBFE edge between the same two components would buy a
    cycle at the price of two more absolute calculations -- and the mode that pays for
    counterpoised cycle coverage is ``cbfe_mode="cycles"``, which the planner applies
    afterwards on the merits of each ligand rather than blindly per component pair.

    Parameters
    ----------
    graph : networkx.Graph
        The feasible RBFE graph. Not mutated.
    ligands : Mapping[str, Ligand]
    pool : Mapping[tuple[str, str], float]
        Eligible CBFE pairs and their costs, from :func:`build_cbfe_pool`.

    Returns
    -------
    list[tuple[str, str]]
        Sorted endpoint pairs, in selection order.
    """
    import networkx as nx

    if not pool:
        return []

    components = [set(component) for component in nx.connected_components(graph)]
    if len(components) < 2:
        return []

    partition = {name: index for index, component in enumerate(components) for name in component}
    return select_bridges(partition, ligands, pool, graph=graph, n_per_pair=1)
