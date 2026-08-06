#!/usr/bin/env python
"""Plan a network in a dozen lines, then check it.

Run ``python examples/data/make_conformers.py`` first to build the ligand file.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from rbfenetmap.core.options import NetworkOptions, SoftcorePolicy
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.io.loaders import load_ligands

DATA = Path(__file__).resolve().parent / "data" / "benzamides.sdf"


def main() -> int:
    """Plan and summarise a network over the example series."""
    ligands = load_ligands([DATA])
    network = build_network(
        ligands,
        mapper="mcss-e2",
        scorer="linear",
        planner="mst",
        network_options=NetworkOptions(
            edges_per_ligand=2, min_cycle_coverage=1.0, softcore=SoftcorePolicy(max_softcore_atoms=12)
        ),
    )
    network.validate()

    graph = network.to_networkx()
    print(f"{len(network.ligands)} ligands, {len(network.edges)} edges")
    print(f"connected: {nx.is_connected(graph)}, minimum degree: {min(dict(graph.degree()).values())}")
    for edge in sorted(network.edges, key=lambda e: e.score.total):
        marker = " (repaired)" if edge.repair.applied else ""
        print(f"  {edge.key:26s} cost {edge.score.total:6.3f}{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
