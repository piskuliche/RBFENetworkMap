"""2D depiction of a transformation's common core and soft-core regions.

Ported from ``BuildEdges.draw_softcore`` / ``_draw_molecule``. Renders SVG rather than
raster images so the output embeds directly in an HTML report with no image encoding, no
Pillow dependency, and no loss of legibility when a reader zooms in on a crowded ring.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from rbfenetmap.core.models import Ligand, Transformation

__all__ = ("CORE_COLOR", "SOFTCORE_COLOR", "render_edge_svg", "render_molecule_svg")

#: Soft-core highlight: warm, meaning "this changes".
SOFTCORE_COLOR = (0.96, 0.55, 0.32)
#: Common-core highlight: cool, meaning "this is held fixed".
CORE_COLOR = (0.55, 0.75, 0.92)


def render_molecule_svg(
    mol: Chem.Mol,
    *,
    softcore: Sequence[int] = (),
    core: Sequence[int] = (),
    width: int = 420,
    height: int = 340,
    title: str = "",
    show_indices: bool = False,
) -> str:
    """Return an SVG string of *mol* with its soft-core and core atoms highlighted.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
    softcore, core : Sequence[int], optional
        Atom indices to highlight.
    width, height : int, optional
        Canvas size in pixels.
    title : str, optional
        Caption drawn under the structure.
    show_indices : bool, optional
        Label atoms with their indices, which is what makes a depiction usable for
        debugging a mapping rather than merely looking at it.

    Returns
    -------
    str
        A standalone ``<svg>`` document.
    """
    # Draw a hydrogen-suppressed copy: explicit hydrogens trebles the atom count and
    # makes the substituent that actually changed impossible to see. The index map keeps
    # the highlights pointing at the right atoms.
    drawable = Chem.Mol(mol)
    keep = [a.GetIdx() for a in drawable.GetAtoms() if a.GetAtomicNum() != 1]
    remap = {old: new for new, old in enumerate(keep)}
    drawable = Chem.RWMol(drawable)
    for index in sorted((a.GetIdx() for a in drawable.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
        drawable.RemoveAtom(index)
    drawable = drawable.GetMol()
    try:
        Chem.SanitizeMol(drawable)
    except (Chem.AtomValenceException, Chem.KekulizeException):  # pragma: no cover - defensive
        drawable = Chem.Mol(mol)
        remap = {i: i for i in range(mol.GetNumAtoms())}

    rdDepictor.Compute2DCoords(drawable)

    highlights: list[int] = []
    colors: dict[int, tuple[float, float, float]] = {}
    for index in core:
        if index in remap:
            highlights.append(remap[index])
            colors[remap[index]] = CORE_COLOR
    for index in softcore:
        if index in remap:
            highlights.append(remap[index])
            colors[remap[index]] = SOFTCORE_COLOR

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addAtomIndices = show_indices
    if title:
        drawer.drawOptions().legendFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, drawable, legend=title, highlightAtoms=highlights, highlightAtomColors=colors
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def render_edge_svg(
    edge: Transformation,
    ligands: Mapping[str, Ligand],
    *,
    width: int = 420,
    height: int = 340,
    show_indices: bool = False,
) -> tuple[str, str]:
    """Return the ``(source_svg, target_svg)`` depictions for one transformation."""
    source = ligands[edge.source]
    target = ligands[edge.target]
    return (
        render_molecule_svg(
            source.mol,
            softcore=edge.mapping.sc1,
            core=edge.mapping.cc1,
            width=width,
            height=height,
            title=f"{edge.source} (soft-core {edge.mapping.n_softcore_1})",
            show_indices=show_indices,
        ),
        render_molecule_svg(
            target.mol,
            softcore=edge.mapping.sc2,
            core=edge.mapping.cc2,
            width=width,
            height=height,
            title=f"{edge.target} (soft-core {edge.mapping.n_softcore_2})",
            show_indices=show_indices,
        ),
    )
