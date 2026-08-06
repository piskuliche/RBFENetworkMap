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
    width: int = 520,
    height: int = 420,
    title: str = "",
    show_indices: bool = False,
    show_hydrogens: bool = True,
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
    show_hydrogens : bool, optional
        Draw the explicit hydrogens. On by default because mappings are stated over
        every atom index including hydrogens, so a hydrogen-suppressed picture cannot
        show what the mapping actually did.

    Returns
    -------
    str
        A standalone ``<svg>`` document.

    Notes
    -----
    The depiction is of the molecule exactly as loaded. Nothing here re-sanitizes or
    re-perceives it, because doing so *adds atoms that are not in the input*: dropping
    the explicit hydrogens frees up valence, and the next ``SanitizeMol`` fills it back
    in with implicit hydrogens. Where the input's bond orders are already wrong -- an
    all-single-bond mol2, say -- that invention is silent and large, turning carbonyls
    into alcohols and aromatic rings into saturated ones. Drawing the molecule untouched
    means a wrong picture is always the input's fault and never this function's.
    """
    drawable = Chem.Mol(mol)
    remap = {index: index for index in range(mol.GetNumAtoms())}

    if not show_hydrogens:
        # Suppress hydrogens for a compact skeletal view. `RemoveHs` does the valence
        # bookkeeping itself, so the hydrogens become implicit rather than invented.
        keep = [a.GetIdx() for a in drawable.GetAtoms() if a.GetAtomicNum() != 1]
        try:
            stripped = Chem.RemoveHs(Chem.Mol(drawable), sanitize=False)
        except Exception:  # pragma: no cover - defensive; keep the faithful drawing
            stripped = None
        if stripped is not None and stripped.GetNumAtoms() == len(keep):
            drawable = stripped
            remap = {old: new for new, old in enumerate(keep)}

    # Replace the 3D conformer with a 2D layout; a molecule drawn straight from
    # crystal-like coordinates is unreadable on a flat page.
    drawable.RemoveAllConformers()
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

    # `PrepareAndDrawMolecule` would add hydrogens to unspecified stereocentres
    # (`addChiralHs` defaults on) -- more atoms that are not in the input. Prepare
    # explicitly with that off, and fall back to an unkekulized drawing rather than
    # letting a molecule with bad bond orders take the whole report down.
    try:
        prepared = rdMolDraw2D.PrepareMolForDrawing(drawable, addChiralHs=False, kekulize=True)
    except Exception:  # pragma: no cover - defensive
        prepared = rdMolDraw2D.PrepareMolForDrawing(drawable, addChiralHs=False, kekulize=False)
    drawer.DrawMolecule(prepared, legend=title, highlightAtoms=highlights, highlightAtomColors=colors)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def render_edge_svg(
    edge: Transformation,
    ligands: Mapping[str, Ligand],
    *,
    width: int = 520,
    height: int = 420,
    show_indices: bool = False,
    show_hydrogens: bool = True,
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
            show_hydrogens=show_hydrogens,
        ),
        render_molecule_svg(
            target.mol,
            softcore=edge.mapping.sc2,
            core=edge.mapping.cc2,
            width=width,
            height=height,
            title=f"{edge.target} (soft-core {edge.mapping.n_softcore_2})",
            show_indices=show_indices,
            show_hydrogens=show_hydrogens,
        ),
    )
