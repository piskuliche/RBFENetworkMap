"""Self-contained HTML report for a planned network.

Everything is inlined -- SVG, CSS, no scripts, no external assets -- so the file can be
emailed, attached to a ticket, or opened from a scratch directory years later and still
render. That constraint is why the depictions are SVG rather than linked images.

The report includes the rejected candidates. A reviewer's first question about a planned
network is almost always "why isn't ligand X connected to Y", and the answer only exists
in the rejections.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from rbfenetmap.viz.depict import render_edge_svg
from rbfenetmap.viz.network_svg import render_network_svg

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rbfenetmap.core.models import Network

__all__ = ("render_report",)

_CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, sans-serif; margin: 0 auto; max-width: 1100px;
       padding: 2rem 1.25rem; line-height: 1.55; }
h1, h2 { line-height: 1.2; }
.intro { max-width: 70ch; }
.summary { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1.5rem 0; }
.stat { border: 1px solid #8883; border-radius: 8px; padding: .75rem 1rem; min-width: 8rem; }
.stat .value { font-size: 1.6rem; font-weight: 600; }
.stat .label { font-size: .8rem; opacity: .75; text-transform: uppercase; letter-spacing: .04em; }
.edge-index { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: .75rem; margin: 1rem 0 1.5rem; }
.edge-index a { display: block; text-decoration: none; color: inherit; border: 1px solid #8883; border-radius: 8px; padding: .8rem .9rem; }
.edge-index a:hover { border-color: #4a90d9; box-shadow: 0 0 0 2px #4a90d922; }
.edge-index strong { display: block; font-family: ui-monospace, monospace; margin-bottom: .25rem; }
.edge { border: 1px solid #8883; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
.edge h3 { margin: 0 0 .5rem; font-family: ui-monospace, monospace; }
.edge:target { border-color: #4a90d9; box-shadow: 0 0 0 3px #4a90d922; scroll-margin-top: 1rem; }
.panes { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; }
.panes svg { max-width: 100%; height: auto; border: 1px solid #8882; border-radius: 6px; }
.meta { font-size: .9rem; opacity: .85; }
.trace { font-family: ui-monospace, monospace; font-size: .8rem; white-space: pre-wrap;
         background: #8881; padding: .6rem; border-radius: 6px; margin-top: .6rem; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #8883; }
th { font-weight: 600; }
.legend span { display: inline-block; margin-right: 1rem; font-size: .85rem; }
.swatch { display: inline-block; width: .8rem; height: .8rem; border-radius: 3px;
          vertical-align: -1px; margin-right: .3rem; }
.warn { border-left: 3px solid #e08a3c; padding-left: .8rem; }
"""


def _escape(value: object) -> str:
    """HTML-escape a value for safe interpolation."""
    return html.escape(str(value))


def _edge_anchor(key: str) -> str:
    """Return a stable fragment id for one edge section."""
    return f"edge-{key}"


def render_report(network: "Network", *, title: str = "RBFE network", show_indices: bool = False) -> str:
    """Return a complete, self-contained HTML document describing *network*.

    Parameters
    ----------
    network : Network
    title : str, optional
    show_indices : bool, optional
        Label atoms with indices in the depictions.

    Returns
    -------
    str
    """
    rejected = network.rejected
    repaired = [e for e in network.edges if e.repair.applied]
    selected_edges = sorted(network.edges, key=lambda e: e.score.total)
    edge_links = {edge.unordered_key: f"#{_edge_anchor(edge.key)}" for edge in selected_edges}

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_escape(title)}</title><style>{_CSS}</style></head><body>",
        f"<h1>{_escape(title)}</h1>",
        "<p class='intro'>Selected edges are shown below with both ligands drawn side by side. "
        "The warm highlight is the soft-core region that changes during the transformation; "
        "the cool highlight is the common core that stays fixed.</p>",
        "<div class='summary'>",
        f"<div class='stat'><div class='value'>{len(network.ligands)}</div><div class='label'>Ligands</div></div>",
        f"<div class='stat'><div class='value'>{len(network.edges)}</div><div class='label'>Edges</div></div>",
        f"<div class='stat'><div class='value'>{len(repaired)}</div><div class='label'>Repaired</div></div>",
        f"<div class='stat'><div class='value'>{len(rejected)}</div><div class='label'>Rejected</div></div>",
        f"<div class='stat'><div class='value'>{_escape(network.planner)}</div><div class='label'>Planner</div></div>",
        "</div>",
    ]

    if network.unmet_constraints:
        parts.append("<div class='warn'><strong>Unmet constraints</strong><ul>")
        parts += [f"<li>{_escape(c)}</li>" for c in network.unmet_constraints]
        parts.append("</ul></div>")

    parts += [
        "<h2>Network</h2>",
        "<div class='scroll'>",
        render_network_svg(network, edge_links=edge_links),
        "</div>",
        "<p class='meta'>Thicker edges are cheaper. Hollow red nodes are unconnected. Click a network edge or the index below to jump to its transformation card.</p>",
        "<h2>Selected edges</h2>",
        "<div class='legend'>",
        "<span><span class='swatch' style='background:#f58c52'></span>soft-core (transformed)</span>",
        "<span><span class='swatch' style='background:#8cbfeb'></span>common core (held fixed)</span>",
        "</div>",
    ]

    if selected_edges:
        parts.append("<div class='edge-index'>")
        for edge in selected_edges:
            parts.append(
                f"<a href='#{_edge_anchor(edge.key)}'><strong>{_escape(edge.key)}</strong>"
                f"<span class='meta'>cost {edge.score.total:.3f} &middot; soft-core "
                f"{edge.mapping.n_softcore_1}/{edge.mapping.n_softcore_2} &middot; "
                f"core {edge.mapping.n_common_core}</span></a>"
            )
        parts.append("</div>")

    for edge in selected_edges:
        source_svg, target_svg = render_edge_svg(edge, network.ligands, show_indices=show_indices)
        parts += [
            f"<div class='edge' id='{_edge_anchor(edge.key)}'>",
            f"<h3>{_escape(edge.key)}</h3>",
            f"<div class='meta'>cost {edge.score.total:.3f} &middot; soft-core "
            f"{edge.mapping.n_softcore_1}/{edge.mapping.n_softcore_2} &middot; common core "
            f"{edge.mapping.n_common_core} &middot; mapper {_escape(edge.mapping.method)}</div>",
            f"<div class='panes scroll'>{source_svg}{target_svg}</div>",
        ]
        if edge.repair.applied:
            trace = "\n".join(edge.repair.trace)
            parts.append(
                f"<div class='meta'>Soft-core repaired: regions {edge.repair.n_fragments_before} "
                f"&rarr; {edge.repair.n_fragments_after}, {edge.repair.n_demoted} atom(s) demoted.</div>"
                f"<div class='trace'>{_escape(trace)}</div>"
            )
        parts.append("</div>")

    if rejected:
        parts += [
            "<h2>Rejected candidates</h2>",
            "<p class='meta'>Why these pairs are not available to the planner.</p>",
            "<div class='scroll'><table><thead><tr><th>Edge</th><th>Reason(s)</th>"
            "<th>Soft-core</th><th>Regions before</th></tr></thead><tbody>",
        ]
        for candidate in sorted(rejected, key=lambda c: c.key):
            reasons = ", ".join(r.value for r in candidate.score.rejections)
            parts.append(
                f"<tr><td>{_escape(candidate.key)}</td><td>{_escape(reasons)}</td>"
                f"<td>{candidate.mapping.n_softcore_1}/{candidate.mapping.n_softcore_2}</td>"
                f"<td>{_escape(candidate.repair.n_fragments_before)}</td></tr>"
            )
        parts.append("</tbody></table></div>")

    parts.append("</body></html>")
    return "".join(parts)
