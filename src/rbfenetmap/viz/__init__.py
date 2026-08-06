"""Visualization: 2D depictions, network diagrams, and HTML reports.

All output is inline SVG or self-contained HTML, so a report has no external assets and
renders anywhere without a plotting stack installed.
"""

from __future__ import annotations

from rbfenetmap.viz.depict import render_edge_svg, render_molecule_svg
from rbfenetmap.viz.gallery import render_report
from rbfenetmap.viz.network_svg import render_network_svg

__all__ = ("render_edge_svg", "render_molecule_svg", "render_network_svg", "render_report")
