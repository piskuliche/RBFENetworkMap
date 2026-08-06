"""Inline SVG rendering of the selected network graph.

Hand-rolled rather than delegated to matplotlib or plotly: the output has to embed in a
self-contained HTML report with no external assets, and a spring layout plus a few
hundred bytes of SVG does that without adding a plotting dependency.
"""

from __future__ import annotations

import math
import html
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rbfenetmap.core.models import Network

__all__ = ("render_network_svg",)


def _layout(graph: nx.Graph, seed: int) -> dict[str, tuple[float, float]]:
    """Return a deterministic spring layout scaled into the unit square."""
    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes)): (0.5, 0.5)}
    positions = nx.spring_layout(graph, seed=seed, weight=None)
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0
    return {node: ((p[0] - min(xs)) / span_x, (p[1] - min(ys)) / span_y) for node, p in positions.items()}


def _edge_href(source: str, target: str, edge_links: dict[tuple[str, str], str] | None) -> str | None:
    """Return the optional hyperlink target for one selected edge."""
    if not edge_links:
        return None
    return edge_links.get(tuple(sorted((source, target))))


def render_network_svg(
    network: "Network",
    *,
    width: int = 760,
    height: int = 560,
    seed: int = 7,
    margin: int = 60,
    edge_links: dict[tuple[str, str], str] | None = None,
) -> str:
    """Return a standalone SVG of the selected network.

    Edge stroke width encodes cost: cheap, high-confidence transformations are drawn
    heavier, so the reliable backbone of the network reads at a glance. Nodes not
    touched by any selected edge are drawn hollow, which makes an unconnected ligand
    visible rather than something the reader has to notice by counting.
    """
    graph = network.to_networkx()
    if graph.number_of_nodes() == 0:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'

    positions = _layout(graph, seed)
    inner_width = width - 2 * margin
    inner_height = height - 2 * margin

    def place(node: str) -> tuple[float, float]:
        """Map layout coordinates onto the canvas."""
        x, y = positions[node]
        return margin + x * inner_width, margin + (1.0 - y) * inner_height

    costs = [data["weight"] for _, _, data in graph.edges(data=True)]
    low, high = (min(costs), max(costs)) if costs else (0.0, 1.0)
    span = (high - low) or 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui, sans-serif">',
        "<style>"
        ".edge{stroke:#7f8fa6;stroke-linecap:round}"
        ".node{fill:#4a90d9;stroke:#22384f;stroke-width:1.5}"
        ".node-isolated{fill:#ffffff;stroke:#c0392b;stroke-width:2}"
        ".label{font-size:12px;fill:#1c2733;text-anchor:middle}"
        "</style>",
    ]

    for source, target, data in graph.edges(data=True):
        x1, y1 = place(source)
        x2, y2 = place(target)
        # Heavier stroke for a cheaper edge.
        stroke = 1.0 + 3.5 * (1.0 - (data["weight"] - low) / span)
        line = (
            f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke-width="{stroke:.2f}"><title>{source} - {target}: cost {data["weight"]:.3f}</title></line>'
        )
        href = _edge_href(source, target, edge_links)
        if href:
            line = f'<a href="{html.escape(href, quote=True)}" aria-label="{html.escape(source)} to {html.escape(target)}">{line}</a>'
        parts.append(line)

    for node in graph.nodes:
        x, y = place(node)
        isolated = graph.degree(node) == 0
        css = "node-isolated" if isolated else "node"
        radius = 9 + min(6, math.sqrt(graph.degree(node)) * 2)
        parts.append(
            f'<circle class="{css}" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}">'
            f"<title>{node} (degree {graph.degree(node)})</title></circle>"
        )
        parts.append(f'<text class="label" x="{x:.1f}" y="{y + radius + 14:.1f}">{node}</text>')

    parts.append("</svg>")
    return "".join(parts)
