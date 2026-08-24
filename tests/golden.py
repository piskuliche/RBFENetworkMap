"""The v0.4.0 selection baseline, and the fingerprint the whole epic is checked against.

Every phase of the network-knobs work adds options that default to today's behaviour. That
claim is only worth something if it is *checked*, and checked against something captured
before any of it landed -- which is what ``tests/data/golden_benzamides.json`` is.

The input is pinned too
-----------------------
``tests/data/golden_benzamides.sdf`` is tracked, rather than the series being read from
``examples/data/benzamides.sdf``. That file is gitignored and regenerated on demand, so it
is absent from a fresh clone and from CI. It is also the wrong kind of input for a baseline
even when present: its coordinates come from a constrained embedding, so an RDKit upgrade
could shift them, move ``core_rmsd``, and change every edge cost -- reporting a planner
regression that is really an input change. A golden test's input has to be pinned as firmly
as its expected output.

What is compared, and what deliberately is not
----------------------------------------------
Not the serialized network. A raw ``network_to_dict`` comparison would be byte-fragile in
exactly the wrong way: every phase adds a ``NetworkOptions`` field, every new field lands in
the options block, and the golden file would need regenerating each time -- which is the same
as not having one, because a baseline you rewrite whenever it disagrees with you checks
nothing.

What must not change is the *selection outcome*: which edges the planner chose, in which
direction, at what cost, and the shape that results. :func:`network_fingerprint` captures
exactly that and nothing else, so it stays stable while the option surface grows and still
fails loudly if a refactor moves a single edge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import networkx as nx

__all__ = ("GOLDEN_DIR", "load_golden", "network_fingerprint", "write_golden")

GOLDEN_DIR = Path(__file__).parent / "data"


def _cycle_covered(graph: "nx.Graph") -> list[str]:
    """Nodes lying on a cycle, by the planner's own definition.

    A node lies on a cycle exactly when it belongs to a biconnected component of two or
    more edges. This mirrors ``MSTRedundancyPlanner._close_cycles.covered`` deliberately:
    a fingerprint that measured coverage differently from the planner would drift from it
    silently.
    """
    nodes: set[str] = set()
    for component in nx.biconnected_component_edges(graph):
        edges = list(component)
        if len(edges) >= 2:
            for u, v in edges:
                nodes.add(u)
                nodes.add(v)
    return sorted(nodes)


def network_fingerprint(network: Any) -> dict[str, Any]:
    """Reduce a planned network to the facts that must not change.

    Costs are rounded to nine decimals. The scorers are pure float arithmetic over
    descriptor sums, so the low bits are reproducible in practice, but pinning them exactly
    would turn an unrelated numpy or RDKit upgrade into a failure of *this* test -- which
    would train the reader to regenerate the baseline, and that is the one habit it cannot
    survive.
    """
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(network.ligands)
    graph.add_edges_from(edge.unordered_key for edge in network.edges)

    degrees = sorted((int(d) for _, d in graph.degree()), reverse=True)
    covered = _cycle_covered(graph)
    connected = graph.number_of_nodes() > 0 and nx.is_connected(graph)

    return {
        "planner": network.planner,
        "ligands": sorted(network.ligands),
        "edges": [
            {"key": edge.key, "kind": edge.kind.value, "cost": round(float(edge.score.total), 9)}
            for edge in network.edges
        ],
        "n_edges": len(network.edges),
        "degree_sequence": degrees,
        "connected": connected,
        "diameter": int(nx.diameter(graph)) if connected else None,
        "cycle_covered": covered,
        "cycle_coverage": round(len(covered) / graph.number_of_nodes(), 9) if graph.number_of_nodes() else 0.0,
        "unmet_constraints": list(network.unmet_constraints),
    }


def load_golden(name: str) -> dict[str, Any]:
    """Read a golden fingerprint by bare name."""
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text())


def write_golden(name: str, fingerprint: Mapping[str, Any]) -> Path:
    """Write a golden fingerprint. Used to capture a baseline, never by a test."""
    path = GOLDEN_DIR / f"{name}.json"
    path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")
    return path
