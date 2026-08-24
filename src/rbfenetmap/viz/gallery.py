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

from rbfenetmap.core.cost import network_cost_summary
from rbfenetmap.core.diagnostics import summarize
from rbfenetmap.core.models import EdgeKind
from rbfenetmap.viz.depict import render_edge_svg
from rbfenetmap.viz.network_svg import render_network_svg

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rbfenetmap.core.models import Network, Transformation

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
.rule { display: inline-block; width: 1.6rem; border-top: 2px solid #7f8fa6;
        vertical-align: 4px; margin-right: .3rem; }
.rule.cbfe { border-top-color: #7c3aed; }
.badge { display: inline-block; font-size: .7rem; font-weight: 600; letter-spacing: .05em;
         border-radius: 4px; padding: .05rem .35rem; vertical-align: 2px; margin-left: .4rem;
         border: 1px solid #7c3aed; color: #7c3aed; }
.badge.syn { border-style: dashed; border-color: #0f766e; color: #0f766e; }
.node-marker { display: inline-block; width: .8rem; height: .8rem; border-radius: 50%;
               border: 2px dashed #22384f; vertical-align: -1px; margin-right: .3rem; }
.warn { border-left: 3px solid #e08a3c; padding-left: .8rem; }
"""


def _escape(value: object) -> str:
    """HTML-escape a value for safe interpolation."""
    return html.escape(str(value))


def _edge_anchor(key: str) -> str:
    """Return a stable fragment id for one edge section."""
    return f"edge-{key}"


def _badge(edge: "Transformation") -> str:
    """Return a CBFE marker for an edge heading, or nothing for an RBFE edge.

    Only the exceptional kind is labelled. Badging both would double the visual noise on a
    report where all but a handful of edges are relative.
    """
    return "<span class='badge'>CBFE</span>" if edge.kind is EdgeKind.CBFE else ""


def _is_synthetic(network: "Network", name: str) -> bool:
    """Whether *name* is a ligand this package invented.

    Tolerant of a name the network does not carry and of ``provenance=None``, because a
    report is the last thing that should refuse to render.
    """
    ligand = network.ligands.get(name)
    return ligand is not None and ligand.synthetic


def _synthetic_badge(edge: "Transformation", network: "Network") -> str:
    """Return a ``SYN`` marker when either endpoint of *edge* was invented.

    A dashed border on the badge, matching the dashed node outline in the network diagram,
    and the three letters carry the meaning on their own. That follows the rule the CBFE
    marker already sets: the exceptional thing is *labelled*, never merely coloured, so it
    survives a greyscale print and a reader who does not distinguish the hues.
    """
    invented = [name for name in (edge.source, edge.target) if _is_synthetic(network, name)]
    if not invented:
        return ""
    return f"<span class='badge syn' title='invented: {_escape(', '.join(invented))}'>SYN</span>"


def _edge_summary(edge: "Transformation") -> str:
    """Describe an edge's atom partition in the terms its kind actually uses.

    A CBFE edge has an empty common core by construction, so reporting "common core 0"
    would read as a broken RBFE edge rather than a correct counterpoised one. Say what is
    happening instead: both ligands decoupled in full.
    """
    if edge.kind is EdgeKind.CBFE:
        return f"{edge.mapping.n_atoms_1}/{edge.mapping.n_atoms_2} atoms fully decoupled &middot; no atom mapping"
    return (
        f"soft-core {edge.mapping.n_softcore_1}/{edge.mapping.n_softcore_2} &middot; "
        f"common core {edge.mapping.n_common_core}"
    )


#: The seed the report's robustness estimate is run with. Fixed rather than exposed: two
#: renders of the same network file have to produce the same document, or the HTML report
#: stops being something you can diff or attach to a ticket. A user who wants to vary it is
#: asking a research question, and ``rbfenet diagnose --seed`` is where that lives.
_REPORT_SEED = 0


def _diagnostics_section(network: "Network") -> list[str]:
    """Render the network-level metrics table.

    The report used to be a picture with no numbers beside it, which left the reader to
    eyeball whether a network was well shaped. These are the numbers that answer that:
    diameter, degree spread, short cycles, and what survives a failed edge.
    """
    report = summarize(network, seed=_REPORT_SEED)
    degrees, robustness, budget = report["degrees"], report["robustness"], report["budget"]
    totals = network_cost_summary(network)

    rows = [
        ("Total cost", f"{totals['score']:.3f} scorer units"),
        ("Estimated machine time", f"{totals['gpu_hours']:.1f} GPU-hours (about ${totals['price']:.2f})"),
        ("Mean cost per edge", f"{report['efficiency']:.3f}"),
        ("Degree", f"min {degrees.minimum} &middot; mean {degrees.mean:.2f} &middot; max {degrees.maximum}"),
        ("Isolated ligands", ", ".join(degrees.isolated) or "none"),
        ("Diameter", "disconnected" if report["diameter"] is None else str(report["diameter"])),
        ("Short cycles", f"{report['n_cycles']} of length &le; {report['max_cycle_length']}"),
        (
            "Robustness",
            f"{robustness.connected_fraction:.0%} of {robustness.n_repeats} trials stay connected at "
            f"{robustness.failure_rate:.0%} edge failure; {robustness.mean_ligands_retained:.1f} ligands "
            "retained on average",
        ),
    ]
    parts = [
        "<h2>Diagnostics</h2>",
        "<p class='meta'>Network-level metrics. Machine time is estimated from published per-edge "
        "measurements and is a report only -- it played no part in choosing these edges.</p>",
        "<div class='scroll'><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>",
    ]
    parts += [f"<tr><td>{_escape(label)}</td><td>{value}</td></tr>" for label, value in rows]
    parts.append("</tbody></table></div>")
    parts.append(f"<p class='meta'>{_escape(budget.message)}</p>")
    return parts


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
    cbfe_edges = network.cbfe_edges
    synthetic = network.synthetic_ligands
    selected_edges = sorted(network.edges, key=lambda e: e.score.total)
    edge_links = {edge.unordered_key: f"#{_edge_anchor(edge.key)}" for edge in selected_edges}

    intro = (
        "<p class='intro'>Selected edges are shown below with both ligands drawn side by side. "
        "The warm highlight is the soft-core region that changes during the transformation; "
        "the cool highlight is the common core that stays fixed."
    )
    if cbfe_edges:
        intro += (
            " Counterpoised (CBFE) edges are marked separately: they run two absolute calculations in "
            "opposite directions rather than morphing one ligand into the other, so both molecules are "
            "entirely soft-core and there is no common core to show."
        )
    if synthetic:
        intro += (
            " Vertices marked <span class='badge syn'>SYN</span>, and drawn with a dashed outline in the "
            "network diagram, are molecules this package <em>invented</em> to bridge a pair no mapping could "
            "relate. Nobody supplied them and nobody has measured them: each one needs parameterising before "
            "any edge touching it can run, and their provenance is listed below."
        )
    intro += "</p>"

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_escape(title)}</title><style>{_CSS}</style></head><body>",
        f"<h1>{_escape(title)}</h1>",
        intro,
        "<div class='summary'>",
        # Real ligands, not all of them. "12 ligands" quietly meaning nine real ones and
        # three inventions is the single most expensive misreading this report can invite,
        # so the invented ones get their own stat rather than being folded into the total.
        f"<div class='stat'><div class='value'>{len(network.ligands) - len(synthetic)}</div>"
        "<div class='label'>Ligands</div></div>",
        f"<div class='stat'><div class='value'>{len(network.edges)}</div><div class='label'>Edges</div></div>",
    ]
    if synthetic:
        parts.append(
            f"<div class='stat'><div class='value'>{len(synthetic)}</div><div class='label'>Invented</div></div>"
        )
    if cbfe_edges:
        parts.append(
            f"<div class='stat'><div class='value'>{len(cbfe_edges)}</div><div class='label'>CBFE edges</div></div>"
        )
    parts += [
        f"<div class='stat'><div class='value'>{len(repaired)}</div><div class='label'>Repaired</div></div>",
        f"<div class='stat'><div class='value'>{len(rejected)}</div><div class='label'>Rejected</div></div>",
        f"<div class='stat'><div class='value'>{_escape(network.planner)}</div><div class='label'>Planner</div></div>",
        "</div>",
    ]

    if network.unmet_constraints:
        parts.append("<div class='warn'><strong>Unmet constraints</strong><ul>")
        parts += [f"<li>{_escape(c)}</li>" for c in network.unmet_constraints]
        parts.append("</ul></div>")

    parts += _diagnostics_section(network)

    parts += [
        "<h2>Network</h2>",
        "<div class='scroll'>",
        render_network_svg(network, edge_links=edge_links),
        "</div>",
        "<p class='meta'>Thicker edges are cheaper. Hollow red nodes are unconnected. "
        + ("Violet edges are counterpoised (CBFE). " if cbfe_edges else "")
        + "Click a network edge or the index below to jump to its transformation card.</p>",
        "<h2>Selected edges</h2>",
        "<div class='legend'>",
        "<span><span class='swatch' style='background:#f58c52'></span>soft-core (transformed)</span>",
        "<span><span class='swatch' style='background:#8cbfeb'></span>common core (held fixed)</span>",
    ]
    if cbfe_edges:
        parts += [
            "<span><span class='rule'></span>RBFE edge</span>",
            "<span><span class='rule cbfe'></span>CBFE edge (counterpoised)</span>",
        ]
    if synthetic:
        parts.append("<span><span class='node-marker'></span>invented ligand (<strong>SYN</strong>)</span>")
    parts.append("</div>")

    if selected_edges:
        parts.append("<div class='edge-index'>")
        for edge in selected_edges:
            parts.append(
                f"<a href='#{_edge_anchor(edge.key)}'>"
                f"<strong>{_escape(edge.key)}{_badge(edge)}{_synthetic_badge(edge, network)}</strong>"
                f"<span class='meta'>cost {edge.score.total:.3f} &middot; {_edge_summary(edge)}</span></a>"
            )
        parts.append("</div>")

    for edge in selected_edges:
        source_svg, target_svg = render_edge_svg(edge, network.ligands, show_indices=show_indices)
        parts += [
            f"<div class='edge' id='{_edge_anchor(edge.key)}'>",
            f"<h3>{_escape(edge.key)}{_badge(edge)}{_synthetic_badge(edge, network)}</h3>",
            f"<div class='meta'>cost {edge.score.total:.3f} &middot; {_edge_summary(edge)} &middot; "
            f"{'protocol' if edge.kind is EdgeKind.CBFE else 'mapper'} {_escape(edge.mapping.method)}</div>",
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

    if synthetic:
        parts += [
            "<h2>Invented ligands</h2>",
            "<p class='meta'>Molecules this package proposed and posed against their parents. "
            "<strong>Pose RMSD</strong> is the in-place deviation of the posed atoms from the parent "
            "coordinates they were taken from, measured with the same function the feasibility gate uses -- "
            "so it is directly comparable to the core-RMSD threshold, and a value close to it is a pose that "
            "only just qualified. <strong>Pose method</strong> says whether the generator handed over a "
            "complete atom correspondence or one had to be recovered by an MCS search; the latter is the "
            "weaker provenance and is named rather than hidden.</p>",
            "<div class='scroll'><table><thead><tr><th>Ligand</th><th>Parents</th><th>Generator</th>"
            "<th>Pose method</th><th>Pose RMSD (&#8491;)</th></tr></thead><tbody>",
        ]
        for ligand in synthetic:
            provenance = ligand.provenance
            parts.append(
                f"<tr><td>{_escape(ligand.name)}<span class='badge syn'>SYN</span></td>"
                f"<td>{_escape(' + '.join(provenance.parents))}</td>"
                f"<td>{_escape(provenance.generator)}</td>"
                f"<td>{_escape(provenance.pose_method)}</td>"
                f"<td>{provenance.pose_rmsd:.3f}</td></tr>"
            )
        parts.append("</tbody></table></div>")

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
