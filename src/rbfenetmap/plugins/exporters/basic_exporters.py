"""Format-neutral exporters: JSON, edge list, and GraphML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from rbfenetmap.core.meta.exporters import AbstractExporter
from rbfenetmap.core.models import Network

__all__ = ("EdgeListExporter", "GraphMLExporter", "JSONExporter")


class JSONExporter(AbstractExporter):
    """Write the full network, including rejected candidates, as JSON.

    The package's own round-trippable format. See :mod:`rbfenetmap.io.networkio`.
    """

    name: ClassVar[str] = "json"
    default_suffix: ClassVar[str] = ".json"

    def export(self, network: Network, destination: Path, **options: Any) -> tuple[Path, ...]:
        """Write ``network.json`` (or *destination* itself if it names a file)."""
        from rbfenetmap.io.networkio import dump_network

        destination = Path(destination)
        path = destination / f"network{self.default_suffix}" if destination.is_dir() else destination
        return (dump_network(network, path, indent=int(options.get("indent", 2))),)


class EdgeListExporter(AbstractExporter):
    """Write a plain ``source target cost`` edge list.

    The lowest-common-denominator format, readable by anything including a shell
    pipeline. Deliberately carries no mapping information -- it is for driving a workflow
    that already knows how to build each edge.
    """

    name: ClassVar[str] = "edgelist"
    default_suffix: ClassVar[str] = ".dat"

    def export(self, network: Network, destination: Path, **options: Any) -> tuple[Path, ...]:
        """Write the edge list."""
        destination = Path(destination)
        path = destination / f"edges{self.default_suffix}" if destination.is_dir() else destination
        path.parent.mkdir(parents=True, exist_ok=True)

        separator = str(options.get("separator", " "))
        # `kind` is appended rather than inserted so a consumer reading positional fields
        # keeps reading the same values it always did.
        lines = ["# source target cost n_softcore_1 n_softcore_2 kind"]
        lines += [
            separator.join(
                (
                    edge.source,
                    edge.target,
                    f"{edge.score.total:.6f}",
                    str(edge.mapping.n_softcore_1),
                    str(edge.mapping.n_softcore_2),
                    edge.kind.value,
                )
            )
            for edge in network.edges
        ]
        path.write_text("\n".join(lines) + "\n")
        return (path,)


class GraphMLExporter(AbstractExporter):
    """Write the selected network as GraphML, for Cytoscape, Gephi, and similar."""

    name: ClassVar[str] = "graphml"
    default_suffix: ClassVar[str] = ".graphml"

    def export(self, network: Network, destination: Path, **options: Any) -> tuple[Path, ...]:
        """Write the GraphML file."""
        del options
        import networkx as nx

        destination = Path(destination)
        path = destination / f"network{self.default_suffix}" if destination.is_dir() else destination
        path.parent.mkdir(parents=True, exist_ok=True)

        graph: nx.Graph = nx.Graph()
        for name, ligand in network.ligands.items():
            graph.add_node(name, charge=ligand.charge, n_heavy=ligand.n_heavy)
        for edge in network.edges:
            # GraphML has no container types, so only scalars go on the attributes.
            graph.add_edge(
                edge.source,
                edge.target,
                weight=float(edge.score.total),
                cost=float(edge.score.total),
                n_softcore_1=edge.mapping.n_softcore_1,
                n_softcore_2=edge.mapping.n_softcore_2,
                n_common_core=edge.mapping.n_common_core,
                repaired=bool(edge.repair.applied),
                method=edge.mapping.method,
                kind=edge.kind.value,
            )
        nx.write_graphml(graph, path)
        return (path,)
