"""Bring a ligand set into a common frame.

The rest of this package assumes ligands arrive co-posed, and measures the mapped core's
RMSD *in place* so that a mapping pairing atoms in different parts of the pocket is caught
rather than flattered. That assumption breaks for structures prepared separately -- ligands
set up individually for ABFE runs, then written to mol2 from their own Amber topologies,
each sitting wherever its own simulation box put it. The conformations are real bound poses;
only the frames disagree. Left alone, every candidate edge between them is rejected for
geometry, which is the correct answer to the wrong question.

This module answers the right one. It fits each ligand onto an already-aligned neighbour and
moves it there rigidly, so the in-place core RMSD downstream measures conformational
difference rather than an accident of where each box was centred.

What it cannot do is worth stating as plainly. Rigid alignment recovers a common **frame**,
never a common **conformation**. Each independently relaxed structure keeps its own ring
puckers, exocyclic torsions, and bond-length noise, so a residual core RMSD survives
alignment and should. The per-ligand records returned here are what let a caller tell the
two apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from rbfenetmap.core.kabsch import apply_transform, core_rmsd, rigid_transform
from rbfenetmap.core.mcs import mcs_embeddings, mcs_query
from rbfenetmap.core.models import Ligand
from rbfenetmap.core.options import AlignmentOptions, MappingOptions

__all__ = ("AlignmentResult", "LigandAlignment", "align_ligands", "choose_reference")

logger = logging.getLogger(__name__)

#: Below this, in angstroms, the second-smallest singular value of a centred fit set means
#: its atoms are effectively collinear. See :func:`_fit_mcs`.
_COLLINEAR_TOLERANCE = 0.1


@dataclass(frozen=True)
class LigandAlignment:
    """What alignment did to one ligand, and how well it worked.

    Parameters
    ----------
    name : str
        The ligand this record describes.
    reference : str, optional
        The already-aligned ligand it was fitted onto. ``None`` for the root, which
        defines the frame and is therefore never moved.
    method : {"mcs", "o3a", "reference", "none"}
        How the transform was obtained. ``"reference"`` marks the root; ``"none"`` marks a
        ligand left in its own frame because no usable fit was found.
    n_fit_atoms : int
        How many corresponding atoms the transform was fitted on. Small values deserve
        suspicion even when the RMSD looks good.
    rmsd : float
        RMSD over those atoms *after* the fit, in angstroms.
    ok : bool
        Whether the ligand was moved into the common frame.
    note : str
        Why not, when ``ok`` is false, or a warning about the fit when it is true.
    """

    name: str
    reference: str | None
    method: str
    n_fit_atoms: int
    rmsd: float
    ok: bool
    note: str = ""

    def as_metadata(self) -> dict[str, Any]:
        """Return a JSON-safe record for :attr:`~rbfenetmap.core.models.Ligand.metadata`.

        Returns
        -------
        dict
            Plain types only, so it survives the network JSON round trip unchanged.
        """
        return {
            "reference": self.reference,
            "method": self.method,
            "n_fit_atoms": int(self.n_fit_atoms),
            "rmsd": float(self.rmsd),
            "ok": bool(self.ok),
            "note": self.note,
        }


@dataclass(frozen=True)
class AlignmentResult:
    """The aligned ligands, plus an account of how each one got there.

    Parameters
    ----------
    ligands : tuple[Ligand, ...]
        In the order they were supplied, not the order they were aligned in.
    reference : str
        The ligand whose frame the set now shares.
    records : tuple[LigandAlignment, ...]
        One per ligand, in the same order as *ligands*.
    """

    ligands: tuple[Ligand, ...]
    reference: str
    records: tuple[LigandAlignment, ...]

    @property
    def failures(self) -> tuple[LigandAlignment, ...]:
        """Records for ligands left in their own frame."""
        return tuple(record for record in self.records if not record.ok)

    @property
    def median_rmsd(self) -> float:
        """Median post-fit RMSD over the ligands that were actually moved.

        Returns
        -------
        float
            ``0.0`` when nothing was moved. This is the number to quote when advising on
            ``core_rmsd_threshold``: it is what alignment achieved, not what it hoped for.
        """
        moved = [record.rmsd for record in self.records if record.ok and record.reference is not None]
        return float(np.median(moved)) if moved else 0.0


def choose_reference(ligands: Mapping[str, Ligand], requested: str | None = None) -> str:
    """Pick the ligand whose frame the set should adopt.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
    requested : str, optional
        An explicit choice. Validated against *ligands*.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If *ligands* is empty, or *requested* names a ligand that was not loaded.

    Notes
    -----
    The automatic rule is the ligand with the most heavy atoms, ties broken by name. In a
    congeneric series the largest member usually contains the shared scaffold, so its
    overlap with every partner is large. The name in the sort key is not decoration: it is
    what makes the choice reproducible when the same set is supplied in a different order.
    """
    if not ligands:
        raise ValueError("Cannot choose an alignment reference from an empty ligand set.")
    if requested is not None:
        if requested not in ligands:
            raise ValueError(
                f"Unknown alignment reference {requested!r}. Loaded ligands: {sorted(ligands)}. "
                "The reference must be one of them."
            )
        return requested
    return sorted(ligands.values(), key=lambda ligand: (-ligand.n_heavy, ligand.name))[0].name


def _heavy_coords(ligand: Ligand) -> np.ndarray:
    """Return the ``(n, 3)`` conformer coordinates of *ligand*, all atoms."""
    return np.asarray(ligand.mol.GetConformer().GetPositions(), dtype=float)


def _is_collinear(coords: np.ndarray) -> bool:
    """Whether *coords* span fewer than two dimensions.

    Notes
    -----
    This is the one failure mode the post-fit RMSD cannot see. Three collinear atoms leave
    the rotation about their shared axis completely free, so the fit reports a perfect
    score while the molecule's orientation is arbitrary. A *planar* set is fine -- the
    determinant correction in :func:`~rbfenetmap.core.kabsch.kabsch_rotation` handles the
    remaining mirror ambiguity -- so the test is on the second-smallest singular value, not
    the smallest.
    """
    if coords.shape[0] < 3:
        return True
    singular_values = np.linalg.svd(coords - coords.mean(axis=0), compute_uv=False)
    return bool(singular_values[1] < _COLLINEAR_TOLERANCE)


def _fit_mcs(
    mobile: Ligand, reference: Ligand, options: AlignmentOptions, mapping_options: MappingOptions
) -> tuple[np.ndarray, np.ndarray, int, float, str] | None:
    """Fit *mobile* onto *reference* through their maximum common substructure.

    Returns
    -------
    tuple or None
        ``(rotation, translation, n_fit_atoms, rmsd, note)``, or ``None`` when the two
        share too little to fit on.

    Notes
    -----
    Candidate embeddings are ranked by RMSD *after* superposition, which is the opposite of
    what :meth:`~rbfenetmap.plugins.mappers.mcss_mapper.MCSSMapper._select_pairing` does and
    is right for a different reason. The mapper is choosing a common core, and in a frame
    the ligands already share, in-place deviation is the signal. Here there is no shared
    frame yet -- in-place deviation would just measure how far apart the boxes were -- and
    the question is which correspondence yields the best rigid superposition. A symmetric
    ring taken the wrong way round shows up plainly under that criterion.
    """
    pattern = mcs_query(mobile.mol, reference.mol, mapping_options)
    if pattern is None:
        return None
    matches_mobile, matches_reference = mcs_embeddings(mobile.mol, reference.mol, pattern, mapping_options)
    if not matches_mobile or not matches_reference:
        return None

    coords_mobile = _heavy_coords(mobile)
    coords_reference = _heavy_coords(reference)
    heavy_mobile = set(mobile.heavy_indices)

    budget = max(options.max_matches, 1)
    best: tuple[float, np.ndarray, np.ndarray, list[int]] | None = None
    evaluated = 0
    for match_mobile in matches_mobile:
        for match_reference in matches_reference:
            if evaluated >= budget:
                break
            evaluated += 1
            pairs = dict(zip(match_mobile, match_reference))
            if len(set(pairs.values())) != len(pairs):
                continue
            # Fit on heavy atoms only. Hydrogens outnumber them and turn on rotamer
            # accidents -- a methyl frozen at a different phase in two independent
            # minimisations would drag the whole molecule round to compensate.
            fit_mobile = [index for index in pairs if index in heavy_mobile]
            if len(fit_mobile) < options.min_mcs_atoms:
                continue
            fit_reference = [pairs[index] for index in fit_mobile]
            rotation, translation = rigid_transform(coords_mobile[fit_mobile], coords_reference[fit_reference])
            rmsd = core_rmsd(
                apply_transform(coords_mobile[fit_mobile], rotation, translation), coords_reference[fit_reference]
            )
            if best is None or rmsd < best[0]:
                best = (rmsd, rotation, translation, fit_reference)
        if evaluated >= budget:
            break

    if best is None:
        return None
    rmsd, rotation, translation, fit_reference = best
    note = ""
    if _is_collinear(coords_reference[fit_reference]):
        note = "fitted atoms are nearly collinear; the rotation about their axis is arbitrary"
        logger.warning("%s aligned onto %s on a nearly collinear atom set: %s", mobile.name, reference.name, note)
    return rotation, translation, len(fit_reference), rmsd, note


def _fit_o3a(mobile: Ligand, reference: Ligand) -> tuple[np.ndarray, np.ndarray, int, float, str] | None:
    """Fit *mobile* onto *reference* by Open3DAlign shape overlap.

    Returns
    -------
    tuple or None
        ``(rotation, translation, n_fit_atoms, rmsd, note)``, or ``None`` if neither the
        MMFF nor the Crippen variant can be built.

    Notes
    -----
    Not the default, because O3A optimises a chemical-feature overlap rather than an atom
    correspondence: it will cheerfully flip a symmetric ring or drop a substituent into a
    different subpocket, and it leaves no auditable set of paired atoms behind. It earns its
    place on sets with no substructure large enough for an MCS to bite on.

    ``GetO3A`` needs MMFF94 parameters for both molecules and raises when an element is
    outside that force field's coverage -- boron and most metals. ``GetCrippenO3A`` uses
    Crippen logP/MR contributions instead, which are defined much more broadly, so it is
    tried second rather than not at all.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    probe = Chem.Mol(mobile.mol)
    try:
        alignment = rdMolAlign.GetO3A(prbMol=probe, refMol=reference.mol)
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "MMFF-based O3A is unavailable for %s (%s); falling back to the Crippen variant.", mobile.name, exc
        )
        try:
            alignment = rdMolAlign.GetCrippenO3A(prbMol=probe, refMol=reference.mol)
        except (ValueError, RuntimeError) as fallback_exc:
            logger.warning("Crippen O3A also failed for %s: %s", mobile.name, fallback_exc)
            return None

    matrix = np.asarray(alignment.Trans()[1], dtype=float)
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    matches = alignment.Matches()
    if not matches:
        return None

    coords_mobile = _heavy_coords(mobile)
    coords_reference = _heavy_coords(reference)
    fit_mobile = [int(pair[0]) for pair in matches]
    fit_reference = [int(pair[1]) for pair in matches]
    rmsd = core_rmsd(apply_transform(coords_mobile[fit_mobile], rotation, translation), coords_reference[fit_reference])
    return rotation, translation, len(matches), rmsd, ""


def _moved(ligand: Ligand, rotation: np.ndarray, translation: np.ndarray, record: LigandAlignment) -> Ligand:
    """Return *ligand* rigidly transformed, with its alignment recorded in metadata.

    Notes
    -----
    The molecule is copied rather than transformed in place. :class:`Ligand` is frozen so
    that callers can hold one without it changing underneath them, and mutating the wrapped
    ``Chem.Mol`` would honour that in letter while breaking it in spirit. The copy also
    preserves per-atom properties -- ``_TriposAtomName`` in particular, which
    :mod:`rbfenetmap.io.amber_masks` relies on.

    :func:`dataclasses.replace` rather than :meth:`Ligand.from_mol`: ``from_mol`` takes
    metadata as keyword arguments, so threading the existing mapping through it would
    collide with any user key called ``name`` or ``source``, and it recomputes a formal
    charge that a rigid motion cannot have changed.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    transformed = Chem.Mol(ligand.mol)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    rdMolTransforms.TransformConformer(transformed.GetConformer(), matrix)
    return replace(
        ligand, mol=transformed, metadata=MappingProxyType({**dict(ligand.metadata), "alignment": record.as_metadata()})
    )


def _annotated(ligand: Ligand, record: LigandAlignment) -> Ligand:
    """Return *ligand* unmoved, carrying its alignment record."""
    return replace(ligand, metadata=MappingProxyType({**dict(ligand.metadata), "alignment": record.as_metadata()}))


def _alignment_order(ligands: Mapping[str, Ligand], reference: str, method: str) -> list[tuple[str, str]]:
    """Return ``(ligand, parent)`` pairs in the order they should be aligned.

    Notes
    -----
    For the MCS method this is a maximum spanning tree over Morgan/Tanimoto similarity,
    grown outward from the reference by Prim's algorithm, so each ligand is fitted onto the
    already-aligned ligand it most resembles rather than onto a root it may share very
    little with. The tree costs nothing an MCS pass would not: fingerprint similarity is
    mapping-free, and the walk still performs exactly ``n - 1`` substructure searches.

    Prim from the root is used rather than :func:`networkx.maximum_spanning_tree` because it
    yields the tree and a valid alignment order together, and because its tie-breaking is
    explicit -- ``(-similarity, name)`` -- where the library function's is not guaranteed.

    O3A gets a plain star instead. It needs no shared substructure, which is the whole
    point of choosing it, so a similarity tree buys nothing; chaining O3A fits would only
    compound each one's error with no correspondence to anchor the drift.
    """
    from rbfenetmap.core.pairs import fingerprint_pair_similarities

    others = sorted(name for name in ligands if name != reference)
    if method != "mcs" or not others:
        return [(name, reference) for name in others]

    pairs = [(a, b) for index, a in enumerate(sorted(ligands)) for b in sorted(ligands)[index + 1 :]]
    similarity = fingerprint_pair_similarities(ligands, pairs)

    def score(a: str, b: str) -> float:
        return similarity.get((a, b), similarity.get((b, a), 0.0))

    aligned = [reference]
    order: list[tuple[str, str]] = []
    remaining = set(others)
    while remaining:
        best = min(((-score(candidate, parent), candidate, parent) for candidate in remaining for parent in aligned))
        _, candidate, parent = best
        order.append((candidate, parent))
        aligned.append(candidate)
        remaining.discard(candidate)
    return order


def align_ligands(
    ligands: Sequence[Ligand] | Mapping[str, Ligand],
    *,
    options: AlignmentOptions | None = None,
    mapping_options: MappingOptions | None = None,
) -> AlignmentResult:
    """Bring a ligand set into a common frame.

    Parameters
    ----------
    ligands : Sequence[Ligand] or Mapping[str, Ligand]
        The set to align. Order is preserved in the result.
    options : AlignmentOptions, optional
        Method, reference, and the fit thresholds. Defaults to
        :class:`~rbfenetmap.core.options.AlignmentOptions`.
    mapping_options : MappingOptions, optional
        Supplies the ``FindMCS`` settings, so that alignment maximises the same
        substructure the mapper will later work from.

    Returns
    -------
    AlignmentResult

    Raises
    ------
    ValueError
        If the set is empty, or ``options.reference`` names a ligand that is not in it.

    Notes
    -----
    A ligand that cannot be fitted is left in its own frame, recorded with ``ok=False``, and
    logged -- not raised over. This package's established line is that an infeasible edge is
    data rather than an error, and an unalignable ligand is the same kind of fact: its edges
    will be rejected for geometry, which is now an honest answer rather than a mystery, and
    the rest of the set still gets a usable network.
    """
    options = options or AlignmentOptions()
    mapping_options = mapping_options or MappingOptions(max_matches=options.max_matches)

    ordered = list(ligands.values()) if isinstance(ligands, Mapping) else list(ligands)
    by_name = {ligand.name: ligand for ligand in ordered}
    if len(by_name) != len(ordered):
        raise ValueError("Ligand names must be unique to align a set; two vertices would otherwise collapse into one.")

    reference_name = choose_reference(by_name, options.reference)
    records: dict[str, LigandAlignment] = {
        reference_name: LigandAlignment(
            name=reference_name, reference=None, method="reference", n_fit_atoms=0, rmsd=0.0, ok=True
        )
    }
    aligned: dict[str, Ligand] = {reference_name: by_name[reference_name]}

    for name, parent in _alignment_order(by_name, reference_name, options.method):
        mobile = by_name[name]
        # Fit onto the parent as it now stands, so the transform lands in the common frame
        # rather than in whatever frame the parent arrived in.
        target = aligned.get(parent, by_name[parent])
        fit = (
            _fit_mcs(mobile, target, options, mapping_options) if options.method == "mcs" else _fit_o3a(mobile, target)
        )
        if fit is None:
            note = (
                f"no usable overlap with {parent} of at least {options.min_mcs_atoms} heavy atoms"
                if options.method == "mcs"
                else f"Open3DAlign could not align it onto {parent}"
            )
            logger.warning(
                "Could not align %s: %s. It stays in its own frame, so its edges will be rejected for geometry.",
                name,
                note,
            )
            records[name] = LigandAlignment(
                name=name, reference=parent, method="none", n_fit_atoms=0, rmsd=float("nan"), ok=False, note=note
            )
            aligned[name] = mobile
            continue

        rotation, translation, n_fit_atoms, rmsd, note = fit
        records[name] = LigandAlignment(
            name=name, reference=parent, method=options.method, n_fit_atoms=n_fit_atoms, rmsd=rmsd, ok=True, note=note
        )
        aligned[name] = _moved(mobile, rotation, translation, records[name])

    result_ligands = tuple(
        aligned[ligand.name] if ligand.name != reference_name else _annotated(ligand, records[reference_name])
        for ligand in ordered
    )
    # Ligands left behind still need their record attached; they were stored unmoved above.
    result_ligands = tuple(
        item if "alignment" in item.metadata else _annotated(item, records[item.name]) for item in result_ligands
    )
    return AlignmentResult(
        ligands=result_ligands, reference=reference_name, records=tuple(records[ligand.name] for ligand in ordered)
    )
