"""Minimum spanning tree plus redundancy: the default network planner.

Selection proceeds in two stages, and the order is what makes the connectivity guarantee
hold. First a minimum spanning tree, seeded so that forced edges are already in it, which
spans every ligand whenever the feasible candidate graph is connected. Then a purely
*additive* redundancy pass that raises degrees and closes cycles without ever removing a
tree edge.

Because the second stage only adds, connectivity established in the first stage cannot be
lost. That is also why an ``n_edges`` smaller than ``n_ligands - 1`` is rejected up front
rather than honoured by trimming: trimming would break the guarantee.
"""

from __future__ import annotations

import warnings
from typing import ClassVar, Mapping, Sequence

import networkx as nx

from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.meta.planners import AbstractNetworkPlanner
from rbfenetmap.core.models import EDGE_SEPARATOR, Ligand, Network, Transformation
from rbfenetmap.core.options import NetworkOptions

__all__ = ("MSTRedundancyPlanner",)


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
    """Return *edge* oriented per *direction*."""
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
    graph: nx.Graph, ligands: Mapping[str, Ligand], candidates: Sequence[Transformation]
) -> str:
    """Build an actionable message explaining why the pool is disconnected.

    Names the components, then the best rejected candidate that would have bridged each
    gap along with its reason. "Disconnected" alone tells a user nothing they can act on;
    "these two groups are only joined by edges rejected for soft-core size" tells them
    exactly which knob to loosen.
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

    classes = _charge_classes(ligands)
    if len(classes) > 1:
        summary = {charge: names[:4] for charge, names in sorted(classes.items())}
        lines.append(
            f"  Note: the ligands span several net charges {summary}. If charge_change_policy is "
            "'reject', no edge can cross those classes -- use 'penalize', or plan each class separately."
        )
    lines.append("  Loosen the soft-core budget, or pass require_connected=False to plan each component.")
    return "\n".join(lines)


class MSTRedundancyPlanner(AbstractNetworkPlanner):
    """Select a spanning network, then add redundancy up to the user's targets."""

    name: ClassVar[str] = "mst"

    def plan(
        self, ligands: Mapping[str, Ligand], candidates: Sequence[Transformation], options: NetworkOptions
    ) -> Network:
        """Select the network. See the module docstring for the ordering rationale.

        Raises
        ------
        rbfenetmap.core.exceptions.NetworkPlanError
            If a forced edge is infeasible, ``n_edges`` cannot span the ligands, or the
            feasible pool is disconnected while connectivity is required.
        """
        names = list(ligands)
        options.check_edge_budget(len(names))

        feasible = _best_by_pair(candidates)
        banned = options.banned_pairs
        feasible = {pair: edge for pair, edge in feasible.items() if pair not in banned}

        self._check_forced(options, feasible, candidates)

        graph: nx.Graph = nx.Graph()
        graph.add_nodes_from(names)
        for pair, edge in feasible.items():
            graph.add_edge(pair[0], pair[1], weight=edge.score.total)

        unmet: list[str] = []
        if len(names) > 1 and not nx.is_connected(graph):
            if options.require_connected:
                raise NetworkPlanError(_describe_disconnection(graph, ligands, candidates))
            unmet.append(f"network is disconnected ({nx.number_connected_components(graph)} components)")

        selected = self._spanning_edges(graph, options)
        selected = self._add_redundancy(graph, selected, options, unmet)

        edges = tuple(_orient(feasible[pair], ligands, options.edge_direction) for pair in sorted(selected))
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
    ) -> None:
        """Raise if any forced edge is unavailable.

        A forced edge bypasses *scoring* but not *feasibility*. Silently dropping one
        would hand back a network missing an edge the user explicitly demanded, with no
        indication anything went wrong.
        """
        by_pair = {c.unordered_key: c for c in candidates}
        problems: list[str] = []
        for pair in sorted(options.forced_pairs):
            if pair in feasible:
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
                "budget for these pairs, or remove them from forced_edges."
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
        """
        selected = set(selected)
        available = sorted(
            (tuple(sorted((u, v))) for u, v in graph.edges if tuple(sorted((u, v))) not in selected),
            key=lambda pair: (graph.edges[pair]["weight"], pair),
        )

        def at_cap() -> bool:
            """Whether the edge budget is exhausted."""
            return options.n_edges is not None and len(selected) >= options.n_edges

        if at_cap() and options.n_edges is not None and len(selected) > options.n_edges:
            unmet.append(f"n_edges={options.n_edges} is below the {len(selected)} edges needed to span the ligands")

        # Degree targets.
        degrees = {node: 0 for node in graph.nodes}
        for source, target in selected:
            degrees[source] += 1
            degrees[target] += 1

        target_degree = options.edges_per_ligand
        progress = True
        while progress and not at_cap():
            progress = False
            deficient = {n for n, d in degrees.items() if d < target_degree}
            if not deficient:
                break
            # Prefer an edge that fixes two deficiencies at once over one that fixes one.
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

        # Cycle coverage. A node lies on a cycle exactly when it belongs to a biconnected
        # component of two or more edges. `biconnected_components` alone is the wrong
        # test: a bridge is a biconnected component of a single edge, so its endpoints
        # would be counted as covered when they are not.
        if options.min_cycle_coverage > 0 and not at_cap():
            selected = self._close_cycles(graph, selected, options, unmet)

        return selected

    def _close_cycles(
        self, graph: nx.Graph, selected: set[tuple[str, str]], options: NetworkOptions, unmet: list[str]
    ) -> set[tuple[str, str]]:
        """Add cheap edges until enough ligands lie on a cycle."""
        selected = set(selected)

        def covered(current: set[tuple[str, str]]) -> set[str]:
            """Nodes belonging to a biconnected component with at least two edges."""
            subgraph: nx.Graph = nx.Graph()
            subgraph.add_nodes_from(graph.nodes)
            subgraph.add_edges_from(current)
            nodes: set[str] = set()
            for component in nx.biconnected_component_edges(subgraph):
                edges = list(component)
                if len(edges) >= 2:
                    for u, v in edges:
                        nodes.add(u)
                        nodes.add(v)
            return nodes

        total = graph.number_of_nodes()
        if total < 3:
            return selected  # a cycle needs three vertices

        target = options.min_cycle_coverage * total
        available = sorted(
            (tuple(sorted((u, v))) for u, v in graph.edges if tuple(sorted((u, v))) not in selected),
            key=lambda pair: (graph.edges[pair]["weight"], pair),
        )

        while len(covered(selected)) < target:
            if options.n_edges is not None and len(selected) >= options.n_edges:
                break
            uncovered = set(graph.nodes) - covered(selected)
            chosen = next(
                (p for p in available if p not in selected and (p[0] in uncovered or p[1] in uncovered)), None
            )
            if chosen is None:
                break
            selected.add(chosen)

        achieved = len(covered(selected)) / total if total else 1.0
        if achieved < options.min_cycle_coverage:
            message = (
                f"min_cycle_coverage={options.min_cycle_coverage:.2f} unmet; achieved {achieved:.2f}. "
                "The candidate pool has no further edges that would close a cycle."
            )
            unmet.append(message)
            warnings.warn(message, stacklevel=4)
        return selected
