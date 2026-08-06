"""Molecular graph utilities, including the node-weighted Steiner tree solver.

Everything here operates on :class:`networkx.Graph` objects whose nodes are atom indices.
Only :func:`mol_to_graph`, :func:`ring_systems`, and :func:`hydrogen_parents` touch
RDKit; the rest -- crucially :func:`node_weighted_steiner`, which is the heart of the
soft-core repair -- is pure graph theory and is unit-testable against hand-built graphs
with no chemistry involved.

This generalizes ``cartograph._connected_subsets`` and the ParmEd-based traversal
helpers in ``BuildEdges`` (``_atoms_beyond_bond``, ``_partition_across_atom``) onto a
single graph representation.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import networkx as nx

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit import Chem

__all__ = (
    "component_beyond_bond",
    "connected_components_of",
    "hydrogen_parents",
    "mol_to_graph",
    "node_weighted_steiner",
    "ring_systems",
    "acyclic_branches",
)


def mol_to_graph(mol: "Chem.Mol") -> nx.Graph:
    """Return the bond graph of *mol*.

    Nodes are atom indices carrying ``element`` (atomic number), ``is_ring``, and
    ``degree`` attributes; edges are bonds carrying ``in_ring``.

    Parameters
    ----------
    mol : rdkit.Chem.Mol

    Returns
    -------
    networkx.Graph
    """
    graph: nx.Graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), element=atom.GetAtomicNum(), is_ring=atom.IsInRing(), degree=atom.GetDegree())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), in_ring=bond.IsInRing())
    return graph


def ring_systems(mol: "Chem.Mol") -> tuple[frozenset[int], ...]:
    """Return the SSSR rings of *mol* as frozensets of atom indices.

    Individual rings, not merged fused systems. That is deliberate: the whole-ring
    closure rule in :mod:`rbfenetmap.core.softcore` iterates to a fixpoint, so absorbing
    one ring of a fused system pulls in the shared atoms, which makes the next ring
    intersected-but-not-contained, which absorbs it in turn. Fused systems therefore
    cascade automatically and need no special case -- matching the conservative behaviour
    of ``cartograph._filter_fused_rings`` without duplicating its logic.
    """
    return tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())


def hydrogen_parents(mol: "Chem.Mol") -> dict[int, int]:
    """Map each terminal hydrogen index to its heavy-atom neighbour.

    Bridging hydrogens (degree > 1) are excluded: they have no single parent, and the
    hydrogen-follows-parent rule is not well defined for them.
    """
    parents: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        neighbours = list(atom.GetNeighbors())
        if len(neighbours) == 1:
            parents[atom.GetIdx()] = neighbours[0].GetIdx()
    return parents


def connected_components_of(graph: nx.Graph, nodes: Iterable[int]) -> list[set[int]]:
    """Return the connected components of the subgraph induced by *nodes*.

    Sorted largest first, then by smallest member, so the result is deterministic.

    This is the rdkit-free generalization of ``cartograph._connected_subsets``.
    """
    node_set = set(nodes)
    components = [set(c) for c in nx.connected_components(graph.subgraph(node_set))]
    return sorted(components, key=lambda c: (-len(c), min(c)))


def component_beyond_bond(graph: nx.Graph, keep: int, start: int) -> set[int]:
    """Return every node reachable from *start* without passing through *keep*.

    The graph-native equivalent of ``BuildEdges._atoms_beyond_bond``.

    Note that for a *ring* bond this does not partition the molecule: the traversal wraps
    around the ring and comes back, so the result contains almost everything. Callers
    that want genuine branches must use :func:`acyclic_branches` instead -- see the note
    there.
    """
    blocked = graph.copy()
    blocked.remove_node(keep)
    if start not in blocked:
        return set()
    return set(nx.node_connected_component(blocked, start))


def acyclic_branches(graph: nx.Graph, center: int) -> dict[int, set[int]]:
    """Partition the graph into the branches hanging off *center* by acyclic bonds only.

    Parameters
    ----------
    graph : networkx.Graph
        The bond graph.
    center : int
        The atom whose substituents are being separated.

    Returns
    -------
    dict[int, set[int]]
        ``{neighbour: nodes_in_that_branch}``, one entry per neighbour reached through a
        bond that is not in a ring. Branches are guaranteed disjoint.

    Notes
    -----
    This deliberately differs from ``BuildEdges._partition_across_atom``, which traverses
    across *every* bond partner while blocking only the central atom. For a ring atom
    that traversal wraps around the ring, so each "branch" contains nearly the whole
    molecule and the branches overlap almost completely. The downstream heuristic then
    demotes all-but-the-largest branch, which is to say almost the entire molecule.

    Restricting the partition to acyclic bonds keeps the branches genuinely disjoint. A
    ring atom simply has fewer branches (possibly none), and the whole-ring closure rule
    in :mod:`rbfenetmap.core.softcore` handles the ring itself.
    """
    branches: dict[int, set[int]] = {}
    for neighbour in sorted(graph.neighbors(center)):
        if graph.edges[center, neighbour].get("in_ring", False):
            continue
        branches[neighbour] = component_beyond_bond(graph, center, neighbour)
    return branches


# ---------------------------------------------------------------------------------
# Node-weighted Steiner tree
# ---------------------------------------------------------------------------------


def _node_split_digraph(
    graph: nx.Graph, terminals: Sequence[set[int]], cost: Callable[[int], float]
) -> tuple[nx.DiGraph, list[str]]:
    """Build the node-split digraph that turns node weights into edge weights.

    Each candidate node ``v`` becomes ``v_in --cost(v)--> v_out``; each bond becomes a
    pair of zero-cost arcs between the relevant halves. Each terminal fragment collapses
    to a single zero-cost supernode, so paths may enter and leave a fragment freely
    without paying for atoms that are already soft-core.

    Returns
    -------
    tuple[networkx.DiGraph, list[str]]
        The digraph and the ordered list of supernode names.
    """
    terminal_of: dict[int, int] = {}
    for index, fragment in enumerate(terminals):
        for node in fragment:
            terminal_of[node] = index

    digraph = nx.DiGraph()
    supernodes = [f"T{i}" for i in range(len(terminals))]
    digraph.add_nodes_from(supernodes)

    def endpoints(node: int) -> tuple[str, str]:
        """Return the (in, out) names for *node*, collapsing terminals to a supernode."""
        if node in terminal_of:
            name = supernodes[terminal_of[node]]
            return name, name
        return f"{node}_in", f"{node}_out"

    for node in graph.nodes:
        if node in terminal_of:
            continue
        node_in, node_out = endpoints(node)
        digraph.add_edge(node_in, node_out, weight=float(cost(node)), atom=node)

    for u, v in graph.edges:
        _, u_out = endpoints(u)
        v_in, _ = endpoints(v)
        digraph.add_edge(u_out, v_in, weight=0.0)
        _, v_out = endpoints(v)
        u_in, _ = endpoints(u)
        digraph.add_edge(v_out, u_in, weight=0.0)

    return digraph, supernodes


def _atoms_on_path(digraph: nx.DiGraph, path: Sequence[str]) -> set[int]:
    """Extract the real atom indices traversed by a path in the node-split digraph."""
    atoms: set[int] = set()
    for u, v in zip(path, path[1:]):
        atom = digraph.edges[u, v].get("atom")
        if atom is not None:
            atoms.add(atom)
    return atoms


def _shortest_connector(digraph: nx.DiGraph, source: str, target: str) -> tuple[float, set[int]] | None:
    """Cheapest set of intermediate atoms connecting two supernodes, or ``None``."""
    try:
        length, path = nx.single_source_dijkstra(digraph, source, target, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return float(length), _atoms_on_path(digraph, path)


def node_weighted_steiner(
    graph: nx.Graph, terminals: Sequence[set[int]], cost: Callable[[int], float]
) -> tuple[set[int], bool]:
    """Find a cheap set of nodes connecting every terminal fragment.

    Parameters
    ----------
    graph : networkx.Graph
        The bond graph.
    terminals : Sequence[set[int]]
        Disjoint node sets to be joined. Fewer than two means there is nothing to do.
    cost : Callable[[int], float]
        Cost of recruiting a node. Terminal nodes are never charged for.

    Returns
    -------
    tuple[set[int], bool]
        The nodes to recruit, and whether the result came from the approximate solver.
        The flag is propagated into the repair trace: an approximate bridge is still
        valid, but it is not guaranteed reproducible across networkx versions, and a user
        comparing two runs deserves to know which is which.

    Raises
    ------
    ValueError
        If the terminals cannot be connected at all -- i.e. the molecule itself is
        disconnected, which no valid ligand should be.

    Notes
    -----
    Two terminals is a shortest-path problem and is solved **exactly** by Dijkstra on the
    node-split digraph.

    Three or more terminals is NP-hard, and is solved by iterative cheapest merge: repeatedly
    find the cheapest node-weighted path joining any two of the current components, recruit
    its interior nodes, and merge. This is the classic greedy Steiner heuristic, costing
    ``O(k^2)`` Dijkstra runs for ``k`` fragments -- trivial at molecular scale, where ``k``
    is rarely above five.

    An earlier version enumerated candidate subsets exhaustively for small instances,
    claiming exactness. That was a mistake: "small" was bounded at 25 candidate nodes, and
    ``C(25, 12)`` is 5.2 million subsets, so real ligands (20-32 candidates, 3-4 fragments)
    hung rather than solving. Exponential search is not viable here even at molecular size.

    Ties are broken by sorted node order everywhere, so repeated runs on the same input
    give the same answer regardless of dictionary or set iteration order.
    """
    fragments = [set(t) for t in terminals if t]
    if len(fragments) < 2:
        return set(), False

    digraph, supernodes = _node_split_digraph(graph, fragments, cost)

    # Exact: two terminals is a shortest-path problem.
    if len(fragments) == 2:
        result = _shortest_connector(digraph, supernodes[0], supernodes[1])
        if result is None:
            raise ValueError(
                f"Cannot connect soft-core fragments {[sorted(f) for f in fragments]}: "
                "no path exists between them, so the molecular graph is disconnected."
            )
        return result[1], False

    return _greedy_merge_steiner(graph, fragments, cost), True


def _greedy_merge_steiner(graph: nx.Graph, fragments: Sequence[set[int]], cost: Callable[[int], float]) -> set[int]:
    """Connect *fragments* by repeatedly joining the cheapest pair of components.

    Each round rebuilds the node-split digraph over the current components, so nodes
    recruited earlier are already free and later paths naturally route through them.

    Raises
    ------
    ValueError
        If no pair of components can be joined, meaning the graph is disconnected.
    """
    components = [set(f) for f in fragments]
    recruited: set[int] = set()

    while len(components) > 1:
        # Sorting keeps the component order, and therefore the outcome, deterministic.
        components.sort(key=lambda c: (-len(c), min(c)))
        digraph, supernodes = _node_split_digraph(graph, components, cost)

        best: tuple[float, int, int, set[int]] | None = None
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                found = _shortest_connector(digraph, supernodes[i], supernodes[j])
                if found is None:
                    continue
                length, atoms = found
                key = (length, i, j)
                if best is None or key < (best[0], best[1], best[2]):
                    best = (length, i, j, atoms)

        if best is None:
            raise ValueError(
                f"Cannot connect soft-core fragments {[sorted(c) for c in components]}: "
                "no path exists between them, so the molecular graph is disconnected."
            )

        _, i, j, atoms = best
        recruited |= atoms
        merged = components[i] | components[j] | atoms
        components = [c for k, c in enumerate(components) if k not in (i, j)] + [merged]

    return recruited


def shortest_node_weighted_path(
    graph: nx.Graph, source: int, target: int, cost: Callable[[int], float]
) -> tuple[float, list[int]]:
    """Cheapest path from *source* to *target* charging :func:`cost` for interior nodes.

    A small Dijkstra used by diagnostics and tests; the repair itself goes through
    :func:`node_weighted_steiner`.

    Returns
    -------
    tuple[float, list[int]]
        Total interior cost and the full node path including both endpoints.
    """
    queue: list[tuple[float, int, list[int]]] = [(0.0, source, [source])]
    best: dict[int, float] = {source: 0.0}
    while queue:
        total, node, path = heapq.heappop(queue)
        if node == target:
            return total, path
        if total > best.get(node, float("inf")):
            continue
        for neighbour in sorted(graph.neighbors(node)):
            step = 0.0 if neighbour == target else float(cost(neighbour))
            new_total = total + step
            if new_total < best.get(neighbour, float("inf")):
                best[neighbour] = new_total
                heapq.heappush(queue, (new_total, neighbour, [*path, neighbour]))
    raise ValueError(f"No path from {source} to {target}.")
