"""Inline SVG rendering of the selected network graph.

Hand-rolled rather than delegated to matplotlib or plotly: the output has to embed in a
self-contained HTML report with no external assets, and a spring layout plus a few
hundred bytes of SVG does that without adding a plotting dependency.
"""

from __future__ import annotations

import math
import html
import os
from typing import TYPE_CHECKING, Sequence

import networkx as nx

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rbfenetmap.core.models import Network

__all__ = ("render_network_svg",)


def _normalize(positions: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Rescale a layout into the unit square."""
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0
    return {node: ((p[0] - min(xs)) / span_x, (p[1] - min(ys)) / span_y) for node, p in positions.items()}


def _layout(graph: nx.Graph, seed: int) -> dict[str, tuple[float, float]]:
    """Return a deterministic layout scaled into the unit square.

    Each connected component is laid out on its own and then given its own cell in a
    grid. A single ``spring_layout`` over a disconnected graph piles the components on
    top of one another -- the repulsive term only acts within a component, so nothing
    pushes two components apart and they settle in the same place. A planned network is
    routinely disconnected, so that degenerate case is the normal one here.
    """
    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes)): (0.5, 0.5)}

    # Deterministic order: largest component first, ties broken by name.
    components = sorted(nx.connected_components(graph), key=lambda c: (-len(c), sorted(c)[0]))

    def component_layout(nodes: set[str]) -> dict[str, tuple[float, float]]:
        """Lay out one component in its own unit square."""
        if len(nodes) == 1:
            return {next(iter(nodes)): (0.5, 0.5)}
        sub = graph.subgraph(nodes)
        # `k` is the target edge length; the default packs nodes tighter than labelled
        # circles can tolerate, so ask for more room and iterate longer to use it.
        spring = nx.spring_layout(sub, seed=seed, weight=None, k=3.0 / math.sqrt(len(nodes)), iterations=600)
        return _normalize(spring)

    if len(components) == 1:
        return component_layout(set(graph.nodes))

    # Pack components into rows, giving each a box proportional to sqrt(size). A uniform
    # grid would hand a lone unmapped ligand the same area as a twenty-node cluster,
    # which is what leaves the big component unreadably cramped.
    sizes = [max(math.sqrt(len(c)), 0.75) for c in components]
    row_target = math.sqrt(sum(s * s for s in sizes) * 1.6)

    rows_of: list[list[int]] = [[]]
    row_width = 0.0
    for index, size in enumerate(sizes):
        if rows_of[-1] and row_width + size > row_target:
            rows_of.append([])
            row_width = 0.0
        rows_of[-1].append(index)
        row_width += size

    row_heights = [max(sizes[i] for i in row) for row in rows_of]
    total_height = sum(row_heights)
    total_width = max(sum(sizes[i] for i in row) for row in rows_of)

    positions: dict[str, tuple[float, float]] = {}
    y_cursor = 0.0
    for row, row_height in zip(rows_of, row_heights):
        x_cursor = 0.0
        for index in row:
            box = sizes[index]
            for node, (x, y) in component_layout(set(components[index])).items():
                # Inset within the box so adjacent components cannot touch.
                local_x = x_cursor + box * (0.10 + 0.80 * x)
                local_y = y_cursor + row_height * (0.10 + 0.80 * y)
                positions[node] = (local_x / total_width, local_y / total_height)
            x_cursor += box
        y_cursor += row_height
    return positions


def _edge_href(source: str, target: str, edge_links: dict[tuple[str, str], str] | None) -> str | None:
    """Return the optional hyperlink target for one selected edge."""
    if not edge_links:
        return None
    return edge_links.get(tuple(sorted((source, target))))


def _label_prefix(names: Sequence[str]) -> str:
    """Return the shared name prefix worth dropping from on-canvas labels.

    Ligand sets are commonly named from one series (``binder_jmc2025-1``,
    ``binder_jmc2025-2``, ...), where the shared part is most of the label width and
    none of the information. The full name stays in the tooltip.
    """
    if len(names) < 2:
        return ""
    prefix = os.path.commonprefix(list(names))
    # Only cut at a separator, so a common prefix never eats part of an identifier.
    cut = max((prefix.rfind(c) for c in "_-"), default=-1)
    if cut <= 0:
        return ""
    prefix = prefix[: cut + 1]
    # Pointless if it would leave labels empty or barely shorter.
    if len(prefix) < 4 or any(len(n) - len(prefix) < 1 for n in names):
        return ""
    return prefix


def render_network_svg(
    network: "Network",
    *,
    width: int | None = None,
    height: int | None = None,
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
    # Scale the canvas with the node count instead of fixing it: 48 ligands in the space
    # that suits 8 is what makes nodes and their labels collide.
    node_count = graph.number_of_nodes()
    if width is None:
        width = max(760, min(2200, int(220 * math.sqrt(max(node_count, 1)))))
    if height is None:
        height = int(width * 0.72)
    if node_count == 0:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>'

    positions = _layout(graph, seed)
    prefix = _label_prefix(sorted(graph.nodes))
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
            f"<title>{html.escape(node)} (degree {graph.degree(node)})</title></circle>"
        )
        label = node[len(prefix) :] if prefix and node.startswith(prefix) else node
        parts.append(
            f'<text class="label" x="{x:.1f}" y="{y + radius + 14:.1f}">{html.escape(label)}'
            f"<title>{html.escape(node)}</title></text>"
        )

    parts.append("</svg>")
    return "".join(parts)
