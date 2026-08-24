"""Post-planning network surgery: add a ligand, drop an edge, join two networks.

Nobody plans once. A campaign gains compounds in batches, loses edges when a run fails to
converge, and grows by joining a new series onto one that is already running. Re-planning
from scratch each time is the wrong answer to all three: it discards the mappings that were
already computed, and -- worse -- it silently reshuffles edges that are already set up,
queued, or finished, so the network you get back is not the network you were running.

Everything here therefore *edits*. :class:`~rbfenetmap.core.models.Network` is frozen, so
each function returns a new one, leaving the input untouched. The edges that were already
there keep their identity, their mappings, and their costs; only the requested change and
its consequences are new.

Invariants
----------
Each function validates its result before returning it, so a surgery that would produce an
inconsistent network fails rather than handing one back. Beyond that:

- **Nothing here re-scores an existing edge.** Costs are comparable across a surgery only
  because the untouched edges are literally the same objects.
- **Connectivity is protected by default.** :func:`delete_edge` refuses an edge whose
  removal would split the network, and names the two sides. Deleting one anyway is
  available, spelled out, and recorded on ``unmet_constraints``.
- **A CBFE edge is never spent to satisfy a degree target**, exactly as in the planner. It
  is used only where a relative edge cannot reach at all -- appending a ligand nothing maps
  to, or bridging two components -- and only when ``cbfe_mode`` allows it.

``cyclize_around_component`` deserves a note: Konnektor declares it and raises
``NotImplementedError``. It is implemented here because a ligand that lies on no cycle has
a free energy nothing checks, and after a deletion or an append that is exactly the ligand
you have.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Iterable, Mapping, Sequence

import networkx as nx

from rbfenetmap.core.cbfe import cbfe_cost, make_cbfe_transformation
from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.meta.scorers import AbstractScorer
from rbfenetmap.core.models import (
    EDGE_SEPARATOR,
    EdgeKind,
    Ligand,
    Network,
    Transformation,
    orient_edge,
    parse_edge_key,
)
from rbfenetmap.core.options import MappingOptions, NetworkOptions

__all__ = ("append_ligand", "concatenate_networks", "cyclize_around_component", "delete_edge", "merge_networks")

logger = logging.getLogger(__name__)


def _pair(spec: str | tuple[str, str] | Sequence[str]) -> tuple[str, str]:
    """Normalize an edge specification to a sorted endpoint pair."""
    if isinstance(spec, str):
        source, target = parse_edge_key(spec)
    else:
        items = tuple(spec)
        if len(items) != 2:
            raise ValueError(f"An edge is a pair of ligand names; got {items!r}.")
        source, target = items
    if source == target:
        raise ValueError(f"Edge specification {spec!r} names the same ligand twice.")
    return tuple(sorted((source, target)))  # type: ignore[return-value]


def _options(network: Network) -> NetworkOptions:
    """Return the network's options, or defaults for a network planned without any."""
    return network.options if network.options is not None else NetworkOptions()


def _selected_graph(network: Network, *, without: tuple[str, str] | None = None) -> nx.Graph:
    """The selected edges as a plain graph, optionally with one pair removed."""
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(network.ligands)
    for edge in network.edges:
        if without is not None and edge.unordered_key == without:
            continue
        graph.add_edge(*edge.unordered_key)
    return graph


def _evaluate(
    ligands: Mapping[str, Ligand],
    pairs: Sequence[tuple[str, str]],
    mapper: AbstractMapper | str,
    scorer: AbstractScorer | str,
    mapping_options: MappingOptions | None,
    options: NetworkOptions,
) -> list[Transformation]:
    """Map, repair, and score *pairs*, resolving plugin names on demand.

    The plugin lookups happen inside the call rather than at module import, which is the
    rule the whole registry rests on: ``core`` never imports a backend.
    """
    from rbfenetmap.core.pipeline import evaluate_pairs
    from rbfenetmap.plugins.mappers import create_mapper
    from rbfenetmap.plugins.scorers import create_scorer

    if not pairs:
        return []
    mapper_obj = create_mapper(mapper) if isinstance(mapper, str) else mapper
    scorer_obj = create_scorer(scorer) if isinstance(scorer, str) else scorer
    return evaluate_pairs(ligands, pairs, mapper_obj, scorer_obj, mapping_options or MappingOptions(), options)


def _describe_rejections(candidates: Sequence[Transformation]) -> str:
    """One line per rejected candidate, cheapest-looking first, for an error message."""
    lines = []
    for candidate in candidates:
        if candidate.feasible:
            continue
        reasons = ", ".join(r.value for r in candidate.score.rejections) or "unknown"
        lines.append(f"    {candidate.key}: {reasons}")
    return "\n".join(sorted(lines))


def append_ligand(
    network: Network,
    ligand: Ligand,
    *,
    n_edges: int = 2,
    mapper: AbstractMapper | str = "mcss-e2",
    scorer: AbstractScorer | str = "linear",
    mapping_options: MappingOptions | None = None,
) -> Network:
    """Add *ligand* to *network*, connecting it with its *n_edges* cheapest edges.

    Parameters
    ----------
    network : Network
        The network to extend. Not modified.
    ligand : Ligand
        The new vertex. Its name must not already be in the network.
    n_edges : int, optional
        How many edges to attach it by. Two is the default because one leaves the new
        ligand hanging off a bridge, where its free energy is checked by nothing; the
        second edge is what puts it on a cycle.
    mapper, scorer : AbstractMapper or AbstractScorer or str, optional
        Used to evaluate the new ligand against every existing one. Pass the same ones the
        network was planned with -- costs from two different scorers are not comparable.
    mapping_options : MappingOptions, optional

    Returns
    -------
    Network
        With the new ligand, its new edges, and the new candidates appended to the audit
        trail. Existing edges are untouched.

    Raises
    ------
    ValueError
        If the ligand's name is already present.
    rbfenetmap.core.exceptions.NetworkPlanError
        If nothing in the network maps to the new ligand and ``cbfe_mode`` does not permit
        a counterpoised edge. The message lists why each pair was rejected.

    Notes
    -----
    Only the new ligand's pairs are mapped: appending to an *n*-ligand network costs *n*
    mappings rather than the *n(n+1)/2* a re-plan would.

    A shortfall -- fewer feasible partners than *n_edges* -- is best-effort and lands on
    ``unmet_constraints``, matching how the planner treats ``edges_per_ligand``. Having *no*
    feasible partner is not a shortfall but a failure, because the result would be a
    disconnected network the caller did not ask for.
    """
    if n_edges < 1:
        raise ValueError("n_edges must be at least 1.")
    if ligand.name in network.ligands:
        raise ValueError(
            f"Ligand {ligand.name!r} is already in the network. Names are vertex identifiers; "
            "rename the new ligand, or delete the existing one first."
        )

    options = _options(network)
    ligands = {**dict(network.ligands), ligand.name: ligand}
    banned = options.banned_pairs
    pairs = sorted(pair for pair in (_pair((ligand.name, name)) for name in network.ligands) if pair not in banned)
    candidates = _evaluate(ligands, pairs, mapper, scorer, mapping_options, options)

    feasible = sorted((c for c in candidates if c.feasible), key=lambda c: (c.score.total, c.unordered_key))
    unmet = list(network.unmet_constraints)
    new_edges: list[Transformation] = list(feasible[:n_edges])

    if not new_edges:
        if options.cbfe_mode != "off":
            partner = min(network.ligands, key=lambda name: (cbfe_cost(ligand, network.ligands[name], options), name))
            new_edges.append(make_cbfe_transformation(ligand, network.ligands[partner], options))
            unmet.append(
                f"{ligand.name} has no feasible RBFE partner in the network; attached as a CBFE edge to {partner}"
            )
        else:
            raise NetworkPlanError(
                f"No feasible edge connects {ligand.name!r} to the network:\n"
                f"{_describe_rejections(candidates)}\n"
                "  Loosen the soft-core budget, set cbfe_mode to attach it with a counterpoised edge, "
                "or plan the enlarged set from scratch.",
                rejected=[c for c in candidates if not c.feasible],
            )
    elif len(new_edges) < n_edges:
        unmet.append(
            f"n_edges={n_edges} unmet for appended ligand {ligand.name}: only {len(new_edges)} feasible partner(s)"
        )

    oriented = tuple(orient_edge(edge, ligands, options.edge_direction) for edge in new_edges)
    result = replace(
        network,
        ligands=ligands,
        edges=network.edges + oriented,
        candidates=network.candidates + tuple(candidates),
        unmet_constraints=tuple(unmet),
    )
    result.validate(require_connected=options.require_connected)
    return result


def delete_edge(network: Network, pair: str | tuple[str, str], *, must_stay_connected: bool = True) -> Network:
    """Remove one edge from *network*.

    Parameters
    ----------
    network : Network
        Not modified.
    pair : str or tuple[str, str]
        The edge, as ``"a~b"`` or as a pair of names. Direction is irrelevant -- selection
        is undirected, so ``"b~a"`` names the same edge.
    must_stay_connected : bool, optional
        Refuse the deletion if it would split the network. On by default.

    Returns
    -------
    Network
        Without that edge. The remaining edges keep their identity and costs.

    Raises
    ------
    ValueError
        If the edge is not in the network, or it is a bridge and *must_stay_connected* is
        set. The bridge message names the two groups the edge is the only link between,
        because "that would disconnect the network" alone does not say what to add instead.

    Notes
    -----
    This is the failure path of a campaign: an edge whose λ windows will not converge, or
    whose setup turns out to be wrong. Deleting it does not re-plan -- see
    :func:`rbfenetmap.core.replanning.replan_after_diagnostics` for the version that
    refills the gap from the candidate pool.
    """
    key = _pair(pair)
    match = next((edge for edge in network.edges if edge.unordered_key == key), None)
    if match is None:
        present = sorted(f"{a}{EDGE_SEPARATOR}{b}" for a, b in (edge.unordered_key for edge in network.edges))
        raise ValueError(
            f"Edge {key[0]}{EDGE_SEPARATOR}{key[1]} is not in the network. Selected edges: "
            f"{present[:8]}{'...' if len(present) > 8 else ''}"
        )

    remaining = _selected_graph(network, without=key)
    still_connected = len(network.ligands) < 2 or nx.is_connected(remaining)
    unmet = list(network.unmet_constraints)

    if not still_connected:
        components = [sorted(component) for component in nx.connected_components(remaining)]
        if must_stay_connected:
            sides = " and ".join(f"{c[:5]}{'...' if len(c) > 5 else ''}" for c in components)
            raise ValueError(
                f"Edge {key[0]}{EDGE_SEPARATOR}{key[1]} is a bridge: it is the only link between {sides}. "
                "Deleting it would leave the network disconnected, so the two halves' free energies could no "
                "longer be compared. Attach another edge across the gap first, or pass "
                "must_stay_connected=False if a disconnected network is intended."
            )
        unmet.append(
            f"network is disconnected ({len(components)} components) after deleting {key[0]}{EDGE_SEPARATOR}{key[1]}"
        )

    result = replace(
        network,
        edges=tuple(edge for edge in network.edges if edge.unordered_key != key),
        unmet_constraints=tuple(unmet),
    )
    result.validate(require_connected=False)
    return result


def _same_molecule(a: Ligand, b: Ligand) -> bool:
    """Whether two ligands of the same name are interchangeable across networks.

    Atom *order* is compared, not just composition, and that is the point. Every mapping in
    a network addresses atoms positionally, so two molecules that are chemically identical
    but were read in different orders make the two networks' mappings mean different
    things. Comparing canonical SMILES would call those equal and merge them into a network
    whose edges silently point at the wrong atoms.
    """
    if a.charge != b.charge or a.n_atoms != b.n_atoms:
        return False
    numbers_a = tuple(atom.GetAtomicNum() for atom in a.mol.GetAtoms())
    numbers_b = tuple(atom.GetAtomicNum() for atom in b.mol.GetAtoms())
    return numbers_a == numbers_b


def _merged_ligands(a: Network, b: Network) -> dict[str, Ligand]:
    """Union the two ligand sets, refusing a shared name that is not the same molecule."""
    ligands = dict(a.ligands)
    conflicts: list[str] = []
    for name, ligand in b.ligands.items():
        existing = ligands.get(name)
        if existing is None:
            ligands[name] = ligand
        elif not _same_molecule(existing, ligand):
            conflicts.append(name)
    if conflicts:
        raise ValueError(
            f"Ligand(s) {sorted(conflicts)} appear in both networks under the same name but are not the "
            "same molecule in the same atom order. Mappings address atoms by index, so merging these would "
            "leave edges pointing at the wrong atoms. Rename one side, or re-plan both from a single input."
        )
    return ligands


def _union_edges(a: Network, b: Network) -> tuple[Transformation, ...]:
    """Union two edge lists, keeping the cheaper of any pair selected by both."""
    chosen: dict[tuple[str, str], Transformation] = {}
    for edge in (*a.edges, *b.edges):
        incumbent = chosen.get(edge.unordered_key)
        if incumbent is None or edge.score.total < incumbent.score.total:
            chosen[edge.unordered_key] = edge
    return tuple(chosen[key] for key in sorted(chosen))


def merge_networks(a: Network, b: Network) -> Network:
    """Merge two networks that share at least one ligand.

    Parameters
    ----------
    a, b : Network
        Neither is modified. They must have at least one ligand name in common, and any
        shared name must denote the same molecule in the same atom order.

    Returns
    -------
    Network
        The union of both ligand sets and both edge sets. A pair selected by both networks
        keeps the cheaper of the two edges. *a*'s options are carried through.

    Raises
    ------
    ValueError
        If the two share no ligand -- which is :func:`concatenate_networks`' job, and the
        message says so -- or if a shared name denotes different molecules.

    Notes
    -----
    Sharing a ligand is what makes the result *comparable* rather than merely combined:
    free energies from the two networks are on the same scale only through a path that
    joins them, and a shared vertex is that path. Two networks with several shared ligands
    also gain cycles through them for free, which is why no bridging is done here.

    The result is not re-planned, so it may exceed ``edges_per_ligand`` around the shared
    ligands. That is deliberate: the extra edges already exist and dropping them would
    discard work.
    """
    shared = sorted(set(a.ligands) & set(b.ligands))
    if not shared:
        raise ValueError(
            "The two networks share no ligand, so merging them would produce a disconnected network "
            "whose two halves are on unrelated free-energy scales. Use concatenate_networks to join "
            "disjoint networks with explicit bridge edges."
        )

    ligands = _merged_ligands(a, b)
    options = _options(a)
    unmet = list(dict.fromkeys((*a.unmet_constraints, *b.unmet_constraints)))
    if a.options is not None and b.options is not None and a.options != b.options:
        unmet.append("merged networks were planned under different options; the first network's options are kept")

    result = Network(
        ligands=ligands,
        edges=_union_edges(a, b),
        candidates=_union_candidates(a, b),
        planner=a.planner if a.planner == b.planner else f"{a.planner}+{b.planner}",
        options=options,
        unmet_constraints=tuple(unmet),
    )
    result.validate(require_connected=False)
    graph = _selected_graph(result)
    if len(ligands) > 1 and not nx.is_connected(graph):
        result = replace(
            result,
            unmet_constraints=(
                *result.unmet_constraints,
                f"network is disconnected ({nx.number_connected_components(graph)} components) after the merge",
            ),
        )
        if options.require_connected:
            result.validate(require_connected=True)
    logger.info("Merged networks over %d shared ligand(s): %s", len(shared), shared[:6])
    return result


def _union_candidates(a: Network, b: Network) -> tuple[Transformation, ...]:
    """Union two candidate pools, keeping one entry per unordered pair."""
    chosen: dict[tuple[str, str], Transformation] = {}
    for candidate in (*a.candidates, *b.candidates):
        incumbent = chosen.get(candidate.unordered_key)
        if incumbent is None or (candidate.feasible and not incumbent.feasible):
            chosen[candidate.unordered_key] = candidate
        elif candidate.feasible and incumbent.feasible and candidate.score.total < incumbent.score.total:
            chosen[candidate.unordered_key] = candidate
    return tuple(chosen[key] for key in sorted(chosen))


def concatenate_networks(
    a: Network,
    b: Network,
    *,
    n_bridges: int = 2,
    mapper: AbstractMapper | str = "mcss-e2",
    scorer: AbstractScorer | str = "linear",
    mapping_options: MappingOptions | None = None,
) -> Network:
    """Join two disjoint networks with *n_bridges* new edges.

    Parameters
    ----------
    a, b : Network
        Neither is modified. Their ligand sets must be disjoint.
    n_bridges : int, optional
        How many edges to build across the join. Two by default, for the same reason
        :func:`append_ligand` attaches two: a single bridge is checked by nothing, whereas
        two put the join itself on a cycle and make the relative offset between the two
        halves verifiable.
    mapper, scorer : AbstractMapper or AbstractScorer or str, optional
    mapping_options : MappingOptions, optional

    Returns
    -------
    Network
        The two ligand sets, both edge sets, and the new bridges.

    Raises
    ------
    ValueError
        If the two share a ligand -- :func:`merge_networks` handles that case.
    rbfenetmap.core.exceptions.NetworkPlanError
        If no cross pair is feasible and ``cbfe_mode`` does not permit a counterpoised
        bridge.

    Notes
    -----
    Every cross pair is evaluated, which is ``len(a) * len(b)`` mappings. That is the honest
    cost of finding the *best* join rather than a plausible one, and it is still far below
    re-planning the union.

    Bridges after the first are chosen to land on ligands the earlier bridges did not
    already use. Two bridges sharing an endpoint make that one ligand a single point of
    failure for the whole join, which is most of what the second bridge was bought to avoid.
    """
    if n_bridges < 1:
        raise ValueError("n_bridges must be at least 1.")
    shared = sorted(set(a.ligands) & set(b.ligands))
    if shared:
        raise ValueError(
            f"The two networks share ligand(s) {shared[:6]}, so they are not disjoint and need no bridges. "
            "Use merge_networks, which joins them through the ligands they already have in common."
        )

    options = _options(a)
    ligands = {**dict(a.ligands), **dict(b.ligands)}
    banned = options.banned_pairs
    pairs = sorted(
        pair for pair in (_pair((left, right)) for left in a.ligands for right in b.ligands) if pair not in banned
    )
    candidates = _evaluate(ligands, pairs, mapper, scorer, mapping_options, options)

    feasible = sorted((c for c in candidates if c.feasible), key=lambda c: (c.score.total, c.unordered_key))
    unmet = list(dict.fromkeys((*a.unmet_constraints, *b.unmet_constraints)))
    bridges: list[Transformation] = []
    used: set[str] = set()
    while feasible and len(bridges) < n_bridges:
        chosen = min(
            feasible, key=lambda c: (sum(name in used for name in c.unordered_key), c.score.total, c.unordered_key)
        )
        feasible.remove(chosen)
        bridges.append(chosen)
        used.update(chosen.unordered_key)

    if not bridges:
        if options.cbfe_mode != "off":
            pair = min(pairs, key=lambda p: (cbfe_cost(ligands[p[0]], ligands[p[1]], options), p))
            bridges.append(make_cbfe_transformation(ligands[pair[0]], ligands[pair[1]], options))
            unmet.append(
                f"no feasible RBFE pair joins the two networks; bridged with a CBFE edge "
                f"{pair[0]}{EDGE_SEPARATOR}{pair[1]}"
            )
        else:
            raise NetworkPlanError(
                "No feasible edge joins the two networks:\n"
                f"{_describe_rejections(candidates)}\n"
                "  Loosen the soft-core budget, set cbfe_mode to bridge them with a counterpoised edge, "
                "or keep them as separate networks.",
                rejected=[c for c in candidates if not c.feasible],
            )
    elif len(bridges) < n_bridges:
        unmet.append(f"n_bridges={n_bridges} unmet: only {len(bridges)} feasible cross-network pair(s)")

    oriented = tuple(orient_edge(edge, ligands, options.edge_direction) for edge in bridges)
    result = Network(
        ligands=ligands,
        # Concatenation cannot produce a duplicate pair -- the ligand sets are disjoint -- so
        # the two edge lists are kept in their original order rather than re-collapsed.
        edges=a.edges + b.edges + oriented,
        candidates=_union_candidates(a, b) + tuple(candidates),
        planner=a.planner if a.planner == b.planner else f"{a.planner}+{b.planner}",
        options=options,
        unmet_constraints=tuple(unmet),
    )
    result.validate(require_connected=options.require_connected)
    return result


def _on_a_cycle(graph: nx.Graph) -> set[str]:
    """Nodes belonging to a biconnected component of two or more edges.

    The same definition the planner's cycle coverage uses. ``biconnected_components`` alone
    gives the wrong answer: a bridge is a biconnected component of a single edge, and its
    endpoints would be counted as covered when nothing checks them.
    """
    nodes: set[str] = set()
    for component in nx.biconnected_component_edges(graph):
        edges = list(component)
        if len(edges) >= 2:
            for u, v in edges:
                nodes.add(u)
                nodes.add(v)
    return nodes


def cyclize_around_component(
    network: Network,
    component: Iterable[str] | None = None,
    *,
    max_cycle_size: int | None = None,
    mapper: AbstractMapper | str | None = None,
    scorer: AbstractScorer | str = "linear",
    mapping_options: MappingOptions | None = None,
) -> Network:
    """Add edges until every ligand in *component* lies on a cycle.

    Parameters
    ----------
    network : Network
        Not modified.
    component : Iterable[str], optional
        The ligands to put on cycles. ``None`` means every ligand in the network. A
        connected component's name is the usual argument -- after a
        :func:`concatenate_networks` or a deletion, it is the newly attached or newly
        exposed part that has ligands hanging off bridges.
    max_cycle_size : int, optional
        Ignore closures longer than this. ``None`` takes the network's own
        ``max_cycle_size``.
    mapper : AbstractMapper or str, optional
        When given, pairs inside *component* that were never evaluated are mapped now.
        ``None`` (the default) restricts the search to the candidates the network already
        carries, which costs nothing and is usually enough: the pool from the original plan
        holds far more feasible pairs than the plan selected.
    scorer : AbstractScorer or str, optional
    mapping_options : MappingOptions, optional

    Returns
    -------
    Network
        With the added edges. A ligand that could not be put on a cycle is recorded on
        ``unmet_constraints`` rather than being an error, matching how the planner reports a
        cycle-coverage shortfall.

    Notes
    -----
    A ligand on no cycle has a free energy nothing checks: every path to it runs through a
    bridge, so an error on that bridge moves the ligand's number and shows up nowhere. That
    is why this exists as its own operation rather than as a re-plan -- after an append or a
    deletion, one or two ligands are in exactly that state and the rest of the network is
    fine.

    Konnektor declares the same operation and raises ``NotImplementedError`` for it.

    Candidates are ranked by how many *new* ligands they put on a cycle, then by kind, then
    by cycle length and cost -- the planner's ranking, including its preference for a
    relative edge over a counterpoised one that would buy the same coverage.
    """
    options = _options(network)
    limit = max_cycle_size if max_cycle_size is not None else options.max_cycle_size
    wanted = set(component) if component is not None else set(network.ligands)
    unknown = sorted(wanted - set(network.ligands))
    if unknown:
        raise ValueError(f"Ligand(s) {unknown} are not in the network.")

    selected = {edge.unordered_key for edge in network.edges}
    banned = options.banned_pairs
    pool: dict[tuple[str, str], Transformation] = {}
    for candidate in network.candidates:
        key = candidate.unordered_key
        if not candidate.feasible or key in selected or key in banned:
            continue
        if key[0] not in wanted or key[1] not in wanted:
            continue
        incumbent = pool.get(key)
        if incumbent is None or candidate.score.total < incumbent.score.total:
            pool[key] = candidate

    if mapper is not None:
        missing = sorted(
            pair
            for pair in (_pair((x, y)) for x in sorted(wanted) for y in sorted(wanted) if x < y)
            if pair not in selected and pair not in banned and pair not in pool
        )
        for candidate in _evaluate(network.ligands, missing, mapper, scorer, mapping_options, options):
            if candidate.feasible:
                pool[candidate.unordered_key] = candidate

    graph = _selected_graph(network)
    added: list[Transformation] = []
    while True:
        covered = _on_a_cycle(graph)
        if wanted <= covered:
            break
        ranked = []
        for key, candidate in pool.items():
            try:
                cycle_size = nx.shortest_path_length(graph, key[0], key[1]) + 1
            except nx.NetworkXNoPath:
                continue
            if limit is not None and cycle_size > limit:
                continue
            trial = graph.copy()
            trial.add_edge(*key)
            gained = len((_on_a_cycle(trial) & wanted) - covered)
            if gained <= 0:
                continue
            is_cbfe = float(candidate.kind is EdgeKind.CBFE)
            ranked.append((-gained, is_cbfe, cycle_size, candidate.score.total, key))
        if not ranked:
            break
        chosen = min(ranked)[-1]
        graph.add_edge(*chosen)
        added.append(pool.pop(chosen))

    unmet = list(network.unmet_constraints)
    stranded = sorted(wanted - _on_a_cycle(graph))
    if stranded:
        unmet.append(
            f"{len(stranded)} ligand(s) still lie on no cycle: {stranded[:6]}{'...' if len(stranded) > 6 else ''}"
        )

    result = replace(
        network,
        edges=network.edges + tuple(orient_edge(edge, network.ligands, options.edge_direction) for edge in added),
        unmet_constraints=tuple(unmet),
    )
    result.validate(require_connected=options.require_connected)
    return result
