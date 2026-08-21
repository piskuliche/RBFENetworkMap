"""Minimum spanning tree plus redundancy: the default network planner.

Selection proceeds in two stages, and the order is what makes the connectivity guarantee
hold. First a minimum spanning tree, seeded so that forced edges are already in it, which
spans every ligand whenever the feasible candidate graph is connected. Then a purely
*additive* redundancy pass that raises degrees, closes cycles, and -- when ``max_diameter``
is set -- buys shortcuts, without ever removing a tree edge.

Because the second stage only adds, connectivity established in the first stage cannot be
lost. That is also why an ``n_edges`` smaller than ``n_ligands - 1`` is rejected up front
rather than honoured by trimming: trimming would break the guarantee.

Counterpoised (CBFE) edges
--------------------------
When ``cbfe_mode`` is not ``"off"`` the guarantee gets stronger rather than weaker: a CBFE
edge exists between every pair of ligands, so the feasible pool can no longer be too sparse
to span. What the mode controls is not whether the stages read CBFE edges but *when those
edges enter the graph*, at three ordered points in :meth:`MSTRedundancyPlanner.plan`:

1. forced pairs that only CBFE can supply -- before the spanning pass, so the pre-seed
   loop can find them;
2. the bridges chosen by :func:`~rbfenetmap.core.cbfe.select_cbfe_bridges` -- also before
   the spanning pass, and before the connectivity check, which is what turns a hard
   disconnection failure into a planned network;
3. the rest of the pool -- *after* the spanning pass, so cycle closure can reach it while
   the degree-raising pass cannot.

Expressing eligibility as presence rather than as a predicate threaded through three
methods is what keeps the stages themselves unchanged. The one place the distinction is
read directly is cycle-closure ranking, which prefers an RBFE edge over a CBFE edge that
would buy the same coverage.
"""

from __future__ import annotations

import warnings
from typing import ClassVar, Mapping, Sequence

import networkx as nx

from rbfenetmap.core.cbfe import build_cbfe_pool, make_cbfe_transformation, select_cbfe_bridges
from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.meta.planners import AbstractNetworkPlanner
from rbfenetmap.core.models import EDGE_SEPARATOR, EdgeKind, Ligand, Network, Transformation
from rbfenetmap.core.options import CBFEMode, NetworkOptions

__all__ = ("MSTRedundancyPlanner", "RedundantMSTPlanner")

#: Guards the cost divisor in the diameter pass. A forced or hub-seeded edge can carry a
#: cost of exactly zero, and "reduction per unit cost" has to stay finite for it.
_COST_FLOOR = 1e-9


def _best_by_pair(candidates: Sequence[Transformation]) -> dict[tuple[str, str], Transformation]:
    """Collapse each unordered pair to its cheapest feasible orientation."""
    best: dict[tuple[str, str], Transformation] = {}
    for candidate in candidates:
        if not candidate.feasible:
            continue
        key = candidate.unordered_key
        incumbent = best.get(key)
        if incumbent is None or candidate.score.total < incumbent.score.total:
            best[key] = candidate
    return best


def _orient(edge: Transformation, ligands: Mapping[str, Ligand], direction: str) -> Transformation:
    """Return *edge* oriented per *direction*.

    For a CBFE edge every atom is soft-core, so ``fewer_softcore_first`` degenerates to
    "smaller ligand first". That is still the convention one wants -- the source is the
    molecule being decoupled from the site -- but it is arrived at by a different route
    than the rationale below describes, which is worth knowing before touching this.
    """
    if direction == "lexicographic":
        return edge if edge.source < edge.target else edge.reversed()
    if direction == "heavier_second":
        source, target = ligands[edge.source], ligands[edge.target]
        return edge if source.n_heavy <= target.n_heavy else edge.reversed()
    # "fewer_softcore_first": start from the side that has less to grow, so the
    # transformation builds outward into the larger ligand.
    return edge if edge.mapping.n_softcore_1 <= edge.mapping.n_softcore_2 else edge.reversed()


def _charge_classes(ligands: Mapping[str, Ligand]) -> dict[int, list[str]]:
    """Group ligand names by net formal charge."""
    classes: dict[int, list[str]] = {}
    for name, ligand in ligands.items():
        classes.setdefault(ligand.charge, []).append(name)
    return classes


def _describe_disconnection(
    graph: nx.Graph, ligands: Mapping[str, Ligand], candidates: Sequence[Transformation], *, cbfe_mode: CBFEMode = "off"
) -> str:
    """Build an actionable message explaining why the pool is disconnected.

    Names the components, then the best rejected candidate that would have bridged each
    gap along with its reason. "Disconnected" alone tells a user nothing they can act on;
    "these two groups are only joined by edges rejected for soft-core size" tells them
    exactly which knob to loosen.

    The closing advice depends on *cbfe_mode*. With CBFE bridging enabled, the soft-core
    budget cannot be the cause -- a CBFE bridge does not use one -- so recommending that
    the user loosen it would send them after the wrong knob. Reaching this message at all
    then means every pair that could have crossed a gap was banned, and that is what the
    message says instead. The rejected-candidate listing above stays useful either way: it
    still explains why no *RBFE* bridge was available.
    """
    components = [sorted(c) for c in nx.connected_components(graph)]
    lines = [f"The feasible candidate graph is disconnected: {len(components)} components."]
    for index, component in enumerate(components):
        preview = component[:6]
        lines.append(f"  component {index + 1} ({len(component)}): {preview}{'...' if len(component) > 6 else ''}")

    membership = {name: index for index, component in enumerate(components) for name in component}
    bridges: dict[tuple[int, int], Transformation] = {}
    for candidate in candidates:
        if candidate.feasible:
            continue
        a = membership.get(candidate.source)
        b = membership.get(candidate.target)
        if a is None or b is None or a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in bridges:
            bridges[key] = candidate

    if bridges:
        lines.append("  Rejected candidates that would have bridged these components:")
        for (a, b), candidate in sorted(bridges.items()):
            reasons = ", ".join(r.value for r in candidate.score.rejections) or "unknown"
            lines.append(f"    {candidate.key} (components {a + 1}/{b + 1}): {reasons}")

    if cbfe_mode in ("bridge", "cycles"):
        lines.append(
            f"  cbfe_mode={cbfe_mode!r} is set, so a CBFE edge could have joined these components without "
            "any mapping at all. Reaching this point means every remaining cross-component pair is in "
            "banned_edges -- remove one of those bans, or pass require_connected=False."
        )
        return "\n".join(lines)

    classes = _charge_classes(ligands)
    if len(classes) > 1:
        summary = {charge: names[:4] for charge, names in sorted(classes.items())}
        lines.append(
            f"  Note: the ligands span several net charges {summary}. If charge_change_policy is "
            "'reject', no edge can cross those classes -- use 'penalize', or plan each class separately."
        )
    lines.append(
        "  Loosen the soft-core budget, set cbfe_mode='bridge' to join the components with "
        "counterpoised edges, or pass require_connected=False to plan each component."
    )
    return "\n".join(lines)


class MSTRedundancyPlanner(AbstractNetworkPlanner):
    """Select a spanning network, then add redundancy up to the user's targets."""

    name: ClassVar[str] = "mst"
    supports_cbfe: ClassVar[bool] = True

    def plan(
        self, ligands: Mapping[str, Ligand], candidates: Sequence[Transformation], options: NetworkOptions
    ) -> Network:
        """Select the network. See the module docstring for the ordering rationale.

        Raises
        ------
        rbfenetmap.core.exceptions.NetworkPlanError
            If a forced edge is unavailable, ``n_edges`` cannot span the ligands, or the
            feasible pool is disconnected while connectivity is required.
        """
        names = list(ligands)
        options.check_edge_budget(len(names))

        feasible = _best_by_pair(candidates)
        banned = options.banned_pairs
        feasible = {pair: edge for pair, edge in feasible.items() if pair not in banned}

        # Costs only; the transformations are built for the handful of pairs that survive
        # selection. Excluding the pairs that already have an RBFE edge here, once, is what
        # stops a pair being offered as both kinds and duplicating in the final network.
        cbfe_pool: dict[tuple[str, str], float] = {}
        if options.cbfe_bridges_components:
            cbfe_pool = build_cbfe_pool(ligands, options, exclude=feasible.keys())

        unmet: list[str] = []
        self._check_forced(options, feasible, candidates, cbfe_available=frozenset(cbfe_pool))

        graph: nx.Graph = nx.Graph()
        graph.add_nodes_from(names)
        for pair, edge in feasible.items():
            graph.add_edge(pair[0], pair[1], weight=edge.score.total, kind=EdgeKind.RBFE.value)

        def adopt(pair: tuple[str, str]) -> None:
            """Move one pair out of the pool and onto the graph as a CBFE edge.

            Popping is what stops an already-placed bridge being offered a second time to
            the cycle-closure pool in step (3).
            """
            graph.add_edge(pair[0], pair[1], weight=cbfe_pool.pop(pair), kind=EdgeKind.CBFE.value)

        # (1) Forced pairs that only CBFE can supply. This must happen before
        # _spanning_edges, whose pre-seed loop skips any pair the graph does not already
        # have -- a forced edge would otherwise vanish without a word.
        for pair in sorted(options.forced_pairs & set(cbfe_pool)):
            adopt(pair)
            spec = f"{pair[0]}{EDGE_SEPARATOR}{pair[1]}"
            unmet.append(f"forced edge {spec} has no feasible RBFE mapping; supplied as a CBFE edge")

        # (2) Bridges, before the connectivity check below, which is what lets a pool that
        # would have been a hard failure become a planned network.
        if options.cbfe_bridges_components:
            for pair in select_cbfe_bridges(graph, ligands, cbfe_pool):
                adopt(pair)

        if len(names) > 1 and not nx.is_connected(graph):
            if options.require_connected:
                raise NetworkPlanError(
                    _describe_disconnection(graph, ligands, candidates, cbfe_mode=options.cbfe_mode),
                    rejected=[c for c in candidates if not c.feasible],
                )
            unmet.append(f"network is disconnected ({nx.number_connected_components(graph)} components)")

        selected = self._spanning_edges(graph, options)

        # (3) The rest of the pool, after the spanning tree is fixed. Cycle closure may
        # spend these; the degree-raising pass filters them back out.
        if options.cbfe_closes_cycles:
            for pair in sorted(cbfe_pool):
                graph.add_edge(pair[0], pair[1], weight=cbfe_pool[pair], kind=EdgeKind.CBFE.value)

        selected = self._add_redundancy(graph, selected, options, unmet)

        def chosen(pair: tuple[str, str]) -> Transformation:
            """Materialize the transformation behind a selected pair."""
            edge = feasible.get(pair)
            if edge is not None:
                return edge
            return make_cbfe_transformation(ligands[pair[0]], ligands[pair[1]], options)

        edges = tuple(_orient(chosen(pair), ligands, options.edge_direction) for pair in sorted(selected))
        network = Network(
            ligands=ligands,
            edges=edges,
            candidates=tuple(candidates),
            planner=self.name,
            options=options,
            unmet_constraints=tuple(unmet),
        )
        network.validate(require_connected=options.require_connected)
        return network

    def _check_forced(
        self,
        options: NetworkOptions,
        feasible: Mapping[tuple[str, str], Transformation],
        candidates: Sequence[Transformation],
        *,
        cbfe_available: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        """Raise if any forced edge is unavailable.

        A forced edge bypasses *scoring* but not *feasibility*. Silently dropping one
        would hand back a network missing an edge the user explicitly demanded, with no
        indication anything went wrong.

        Parameters
        ----------
        options : NetworkOptions
        feasible : Mapping[tuple[str, str], Transformation]
            The post-ban RBFE pool.
        candidates : Sequence[Transformation]
            Used to quote the rejection reason for a forced pair that was scored.
        cbfe_available : frozenset[tuple[str, str]], optional
            Pairs a CBFE edge could supply. These satisfy the constraint, so they are not
            problems -- but the caller still records the substitution, because being handed
            a counterpoised calculation where a relative one was demanded is a change the
            user needs to see.
        """
        by_pair = {c.unordered_key: c for c in candidates}
        problems: list[str] = []
        for pair in sorted(options.forced_pairs):
            if pair in feasible or pair in cbfe_available:
                continue
            candidate = by_pair.get(pair)
            spec = f"{pair[0]}{EDGE_SEPARATOR}{pair[1]}"
            if candidate is None:
                problems.append(f"{spec}: never scored (is either ligand in the input?)")
            else:
                reasons = ", ".join(r.value for r in candidate.score.rejections) or "unknown"
                problems.append(f"{spec}: rejected ({reasons})")
        if problems:
            raise NetworkPlanError(
                "Forced edge(s) cannot be used:\n  " + "\n  ".join(problems) + "\nLoosen the soft-core "
                "budget for these pairs, set cbfe_mode to supply them as counterpoised edges, or remove "
                "them from forced_edges."
            )

    def _spanning_edges(self, graph: nx.Graph, options: NetworkOptions) -> set[tuple[str, str]]:
        """Kruskal, with forced edges and any hub star pre-unioned into the forest."""
        selected: set[tuple[str, str]] = set()
        components: dict[str, str] = {node: node for node in graph.nodes}

        def find(node: str) -> str:
            """Union-find with path compression."""
            while components[node] != node:
                components[node] = components[components[node]]
                node = components[node]
            return node

        def union(a: str, b: str) -> bool:
            """Merge two components; ``False`` if they were already joined."""
            root_a, root_b = find(a), find(b)
            if root_a == root_b:
                return False
            components[root_b] = root_a
            return True

        preseed: list[tuple[str, str]] = sorted(options.forced_pairs)
        if options.hub:
            preseed += sorted(pair for pair in graph.edges if options.hub in pair)
        for pair in preseed:
            key = tuple(sorted(pair))
            if graph.has_edge(*key):
                selected.add(key)  # type: ignore[arg-type]
                union(*key)

        for source, target, weight in sorted(graph.edges(data="weight"), key=lambda e: (e[2], e[0], e[1])):
            del weight
            if union(source, target):
                selected.add(tuple(sorted((source, target))))  # type: ignore[arg-type]
        return selected

    def _add_redundancy(
        self, graph: nx.Graph, selected: set[tuple[str, str]], options: NetworkOptions, unmet: list[str]
    ) -> set[tuple[str, str]]:
        """Greedily add cheap edges to raise degrees and close cycles.

        Strictly additive, so the spanning property established upstream survives.

        The two passes are handed *different* pools, and that is the whole of the
        ``cycles`` mode gate. Cycle closure may reach for a CBFE edge, because putting a
        ligand on a cycle is what makes its free energy checkable and there may be no other
        way to do it. Degree raising may not: an extra edge on an already-connected,
        already-cycled ligand is a refinement, and spending two absolute calculations on a
        refinement is not a trade anyone would make deliberately.
        """
        selected = set(selected)
        available = sorted(
            (tuple(sorted((u, v))) for u, v in graph.edges if tuple(sorted((u, v))) not in selected),
            key=lambda pair: (graph.edges[pair]["weight"], pair),
        )
        degree_available = [pair for pair in available if graph.edges[pair]["kind"] != EdgeKind.CBFE.value]

        def at_cap() -> bool:
            """Whether the edge budget is exhausted."""
            return options.n_edges is not None and len(selected) >= options.n_edges

        if at_cap() and options.n_edges is not None and len(selected) > options.n_edges:
            unmet.append(f"n_edges={options.n_edges} is below the {len(selected)} edges needed to span the ligands")

        if options.selection_objective == "connectivity_then_cycles":
            if options.min_cycle_coverage > 0:
                selected = self._close_cycles(graph, available, selected, options, unmet)
            if options.edges_per_ligand > 1:
                selected = self._raise_degrees(graph, degree_available, selected, options, unmet)
        else:
            selected = self._raise_degrees(graph, degree_available, selected, options, unmet)
            if options.min_cycle_coverage > 0:
                selected = self._close_cycles(graph, available, selected, options, unmet)

        # Third and last, because a diameter bound is a statement about the network the
        # other two passes have already produced. Running it earlier would buy shortcuts
        # across a topology that degree raising and cycle closure then shorten anyway.
        if options.max_diameter is not None:
            selected = self._reduce_diameter(graph, degree_available, selected, options, unmet)

        return selected

    def _raise_degrees(
        self,
        graph: nx.Graph,
        available: Sequence[tuple[str, str]],
        selected: set[tuple[str, str]],
        options: NetworkOptions,
        unmet: list[str],
    ) -> set[tuple[str, str]]:
        """Greedily add cheap edges until every ligand reaches the target degree."""
        selected = set(selected)
        degrees = {node: 0 for node in graph.nodes}
        for source, target in selected:
            degrees[source] += 1
            degrees[target] += 1

        target_degree = options.edges_per_ligand
        progress = True
        while progress and (options.n_edges is None or len(selected) < options.n_edges):
            progress = False
            deficient = {n for n, d in degrees.items() if d < target_degree}
            if not deficient:
                break
            ranked = sorted(
                (p for p in available if p not in selected and (p[0] in deficient or p[1] in deficient)),
                key=lambda p: (-((p[0] in deficient) + (p[1] in deficient)), graph.edges[p]["weight"], p),
            )
            if not ranked:
                break
            chosen = ranked[0]
            selected.add(chosen)
            degrees[chosen[0]] += 1
            degrees[chosen[1]] += 1
            progress = True

        remaining_deficient = sorted(n for n, d in degrees.items() if d < target_degree)
        if remaining_deficient:
            message = (
                f"edges_per_ligand={target_degree} unmet for {len(remaining_deficient)} ligand(s): "
                f"{remaining_deficient[:6]}{'...' if len(remaining_deficient) > 6 else ''}"
            )
            unmet.append(message)
            warnings.warn(message, stacklevel=3)
        return selected

    def _reduce_diameter(
        self,
        graph: nx.Graph,
        available: Sequence[tuple[str, str]],
        selected: set[tuple[str, str]],
        options: NetworkOptions,
        unmet: list[str],
    ) -> set[tuple[str, str]]:
        """Buy shortcut edges until the network's diameter meets ``max_diameter``.

        Statistical error accumulates along a path, so a network two ligands can only be
        compared across in nine hops is worse than its edge count suggests. LOMAP caps the
        diameter at 6 and FEP+ below 5.

        Both of those enforce the bound during edge *removal*, where the question is which
        edge to keep. Selection here is additive -- the spanning tree is never trimmed, and
        that is what makes the connectivity guarantee hold -- so the bound is approached
        from the other side, by adding the shortcut that shortens the network most per unit
        of cost, and repeating.

        Best-effort, tier 5 alongside ``edges_per_ligand`` and ``min_cycle_coverage``: a
        pool with no shortcut left to sell warns and records the shortfall. It never raises,
        because a network that is merely longer than asked for is still a usable network,
        unlike one that fails a hard budget conflict.

        Parameters
        ----------
        graph : networkx.Graph
            The candidate graph, carrying ``weight`` per edge.
        available : Sequence[tuple[str, str]]
            Cost-ordered unselected pairs. The caller passes the *RBFE-only* pool, on the
            same reasoning that keeps degree raising off CBFE edges: shortening a path that
            already exists is a refinement, not a rescue, and two absolute calculations is
            not a trade anyone would make for one.
        selected : set[tuple[str, str]]
        options : NetworkOptions
        unmet : list[str]
            Appended to in place when the target cannot be met.

        Returns
        -------
        set[tuple[str, str]]
            *selected* plus whatever shortcuts were bought.

        Notes
        -----
        :func:`networkx.diameter` is called with ``usebounds=True`` throughout. The bounded
        form (the FastLomap optimisation, arXiv:2304.04713) prunes the all-pairs sweep to a
        handful of BFS runs, which is what makes a per-candidate re-evaluation affordable
        past a hundred ligands.
        """
        selected = set(selected)
        target = options.max_diameter
        if target is None:  # pragma: no cover - the caller only enters this pass when it is set
            return selected

        def diameter_of(current: set[tuple[str, str]]) -> int | None:
            """Diameter of the selected subgraph, or ``None`` if it is disconnected."""
            subgraph: nx.Graph = nx.Graph()
            subgraph.add_nodes_from(graph.nodes)
            subgraph.add_edges_from(current)
            if subgraph.number_of_nodes() < 2:
                return 0
            if not nx.is_connected(subgraph):
                return None
            return int(nx.diameter(subgraph, usebounds=True))

        current = diameter_of(selected)
        if current is None:
            # Diameter is undefined across components. Reporting that plainly beats
            # reporting an unmet bound, which would send the reader after the wrong knob:
            # the network is disconnected, which is the larger problem.
            unmet.append(
                f"max_diameter={target} not evaluated: the selected network is disconnected, so its "
                "diameter is undefined. Connect it first."
            )
            return selected

        while current > target:
            if options.n_edges is not None and len(selected) >= options.n_edges:
                break
            best: tuple[float, int, float, tuple[str, str]] | None = None
            for pair in available:
                if pair in selected:
                    continue
                reduced = diameter_of(selected | {pair})
                if reduced is None or reduced >= current:
                    continue
                cost = float(graph.edges[pair]["weight"])
                reduction = current - reduced
                # Reduction per unit cost first, then raw reduction, then price. Ranking on
                # raw reduction alone would happily pay a rescue-priced edge to save the
                # same hop a cheap one saves.
                key = (-reduction / max(cost, _COST_FLOOR), -reduction, cost, pair)
                if best is None or key < best:
                    best = key
            if best is None:
                break
            selected.add(best[-1])
            current = diameter_of(selected)
            if current is None:  # pragma: no cover - adding an edge cannot disconnect
                break

        if current is not None and current > target:
            message = (
                f"max_diameter={target} unmet; achieved {current}. The candidate pool has no "
                "further edge that would shorten the network."
            )
            unmet.append(message)
            warnings.warn(message, stacklevel=4)
        return selected

    def _close_cycles(
        self,
        graph: nx.Graph,
        available: Sequence[tuple[str, str]],
        selected: set[tuple[str, str]],
        options: NetworkOptions,
        unmet: list[str],
    ) -> set[tuple[str, str]]:
        """Add cheap edges until enough of the network lies on a cycle.

        *available* is the cost-ordered pool of unselected pairs, computed once by the
        caller. It includes CBFE edges when ``cbfe_mode`` allows cycle closure to use them.

        ``cycle_coverage_mode`` chooses what "enough" is measured over. The node form is
        LOMAP's: the fraction of *ligands* on at least one cycle. The edge form is FEP+'s:
        the fraction of *selected edges* on one, which is the complement of the bridge set
        and therefore exactly 2-edge-connectivity at coverage 1.0.

        The edge form is strictly harder, which is why it is opt-in. A bridge hanging off a
        cycle has both endpoints covered under the node rule while the edge itself is
        checked by nothing -- and it is the edge that carries the free energy.

        Note that the edge denominator *moves*: each added edge is one more edge that must
        itself end up on a cycle. That is the intended reading rather than an oversight, so
        the ratio is recomputed each pass instead of being fixed once like the node target.
        """
        selected = set(selected)

        def _subgraph(current: set[tuple[str, str]]) -> nx.Graph:
            """Scratch graph over every ligand carrying only the currently selected edges."""
            subgraph: nx.Graph = nx.Graph()
            subgraph.add_nodes_from(graph.nodes)
            subgraph.add_edges_from(current)
            return subgraph

        def covered_nodes(current: set[tuple[str, str]]) -> set[str]:
            """Nodes belonging to a biconnected component with at least two edges."""
            nodes: set[str] = set()
            for component in nx.biconnected_component_edges(_subgraph(current)):
                edges = list(component)
                if len(edges) >= 2:
                    for u, v in edges:
                        nodes.add(u)
                        nodes.add(v)
            return nodes

        def covered_edges(current: set[tuple[str, str]]) -> set[tuple[str, str]]:
            """Selected edges that are not bridges, i.e. that lie on at least one cycle."""
            bridges = {tuple(sorted(pair)) for pair in nx.bridges(_subgraph(current))}
            return {pair for pair in current if pair not in bridges}

        edge_mode = options.cycle_coverage_mode == "edge"
        covered = covered_edges if edge_mode else covered_nodes

        total = graph.number_of_nodes()
        if total < 3:
            return selected  # a cycle needs three vertices

        target = options.min_cycle_coverage * total

        def below_target(current: set[tuple[str, str]]) -> bool:
            """Whether *current* still falls short of the requested coverage.

            The node branch keeps the original comparison verbatim rather than routing
            through a fraction, so adding the edge mode cannot perturb a single node-mode
            selection by a floating-point hair.
            """
            if edge_mode:
                return bool(current) and len(covered_edges(current)) < options.min_cycle_coverage * len(current)
            return len(covered_nodes(current)) < target

        def achieved_coverage(current: set[tuple[str, str]]) -> float:
            """The coverage fraction actually reached, in whichever unit the mode counts."""
            denominator = len(current) if edge_mode else total
            return len(covered(current)) / denominator if denominator else 1.0

        def cycle_size_if_added(current: set[tuple[str, str]], pair: tuple[str, str]) -> int | None:
            """Return the cycle length created by adding *pair*, or ``None`` if none."""
            subgraph: nx.Graph = nx.Graph()
            subgraph.add_nodes_from(graph.nodes)
            subgraph.add_edges_from(current)
            try:
                return nx.shortest_path_length(subgraph, pair[0], pair[1]) + 1
            except nx.NetworkXNoPath:
                return None

        def rank(
            pair: tuple[str, str], current_covered: set[str]
        ) -> tuple[float, float, float, float, tuple[str, str]] | None:
            """Rank a candidate edge by new coverage, kind, shortness, then cost.

            The kind term sits between coverage and cycle size deliberately. Cost already
            carries the CBFE penalty, so this is not about price -- it is about provenance:
            two edges that put the same ligands on a cycle should resolve to the one that
            does not need a second simulation protocol, and an RBFE five-cycle is worth
            more than a counterpoised four-cycle.
            """
            cycle_size = cycle_size_if_added(selected, pair)
            if cycle_size is None:
                return None
            if options.max_cycle_size is not None and cycle_size > options.max_cycle_size:
                return None
            expanded = set(selected)
            expanded.add(pair)
            gained = len(covered(expanded) - current_covered)
            if gained <= 0:
                return None
            is_cbfe = float(graph.edges[pair]["kind"] == EdgeKind.CBFE.value)
            return (-gained, is_cbfe, cycle_size, graph.edges[pair]["weight"], pair)

        while below_target(selected):
            if options.n_edges is not None and len(selected) >= options.n_edges:
                break
            current_covered = covered(selected)
            # The CBFE restriction below is about *ligands* with no other route onto a
            # cycle, so it reads node coverage in both modes. Under the edge mode
            # ``current_covered`` holds edges and would silently never match a name.
            covered_names = covered_nodes(selected) if edge_mode else current_covered
            # A CBFE edge is only worth spending on a ligand that has no other route onto a
            # cycle, so restrict that half of the pool to pairs with an uncovered endpoint.
            # This is the intent stated directly, and it also keeps `rank` -- which builds
            # two scratch graphs per candidate -- off the quadratic CBFE pool.
            considered = [
                pair
                for pair in available
                if pair not in selected
                and (
                    graph.edges[pair]["kind"] != EdgeKind.CBFE.value
                    or pair[0] not in covered_names
                    or pair[1] not in covered_names
                )
            ]
            ranked = [item for item in (rank(pair, current_covered) for pair in considered) if item is not None]
            chosen = min(ranked)[-1] if ranked else None
            if chosen is None:
                break
            selected.add(chosen)

        achieved = achieved_coverage(selected)
        if achieved < options.min_cycle_coverage:
            # The node wording is left exactly as it was: it is the default, it lands on
            # ``unmet_constraints``, and that string is part of the golden fingerprint.
            unit = " (edge coverage)" if edge_mode else ""
            message = (
                f"min_cycle_coverage={options.min_cycle_coverage:.2f}{unit} unmet; achieved {achieved:.2f}. "
                "The candidate pool has no further edges that would improve cycle coverage."
            )
            unmet.append(message)
            warnings.warn(message, stacklevel=4)
        return selected


class RedundantMSTPlanner(MSTRedundancyPlanner):
    """Overlay ``n_redundancy`` spanning trees, then add the usual redundancy.

    A distinct topology from MST-plus-greedy-redundancy, and one that is separately
    benchmarked: Konnektor builds its default network this way with two trees, and the
    paper that introduced it uses three. Running Kruskal, deleting the edges it chose, and
    running it again yields a second-cheapest spanning structure that shares no edge with
    the first, so every ligand has two independent routes into the network rather than a
    tree plus whatever the greedy degree pass happened to find cheap.

    The difference is what fails when an edge fails. Under the greedy pass a ligand's
    second edge may well be its first edge's neighbour; under overlaid trees it is, by
    construction, part of a structure that spans without the first tree at all.

    Everything else -- the CBFE ordering, the redundancy passes, the connectivity
    guarantee -- is inherited unchanged. Only :meth:`_spanning_edges` differs, and it only
    ever adds, so the guarantee still holds.
    """

    name: ClassVar[str] = "redundant-mst"
    supports_cbfe: ClassVar[bool] = True

    def _spanning_edges(self, graph: nx.Graph, options: NetworkOptions) -> set[tuple[str, str]]:
        """Run Kruskal ``n_redundancy`` times, removing each pass's edges before the next.

        Later passes see a thinner graph and will usually return a spanning *forest* rather
        than a tree -- once the cheapest tree is gone, some ligands may have no edge left.
        That is fine and deliberate: the union still spans, because the first pass did, and
        every pass after it is pure addition.

        Forced pairs and any hub star are pre-seeded by the inherited implementation on the
        first pass only. They are gone from the working copy afterwards, so a later pass
        cannot select them twice.
        """
        selected: set[tuple[str, str]] = set()
        working: nx.Graph = graph.copy()
        for _ in range(options.n_redundancy):
            if working.number_of_edges() == 0:
                break
            pass_edges = super()._spanning_edges(working, options)
            if not pass_edges:
                break
            selected |= pass_edges
            working.remove_edges_from(pass_edges)
        return selected
