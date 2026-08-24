"""Posing an invented molecule into its parents' binding-site frame.

Everything else in this package assumes its ligands arrive already posed in a shared
frame -- the in-place :func:`~rbfenetmap.core.kabsch.core_rmsd` gate depends on it. A
molecule the planner invents has no such pose, and has no crystal structure to inherit
one from, so it has to be built. This module is the only place that happens.

Neither of the two obvious tools is right
-----------------------------------------

``AllChem.ConstrainedEmbed`` pins a *scaffold* and re-embeds everything else from
scratch. That throws away the one thing an intermediate uniquely has: every heavy atom of
a hybrid molecule has a *specific* corresponding atom in a parent whose position in the
pocket is already known. Re-embedding from the scaffold outwards lets ETKDG place an
ortho substituent where the parent had it meta -- a valid conformer of the right molecule
in the wrong place, which is exactly the failure the geometry gate is least able to
explain to a user.

:mod:`rbfenetmap.core.align` is the other near-miss. It recovers a common *frame* for
molecules that already have good conformers. An intermediate has none until this module
makes one, so there is nothing for it to align.

The algorithm
-------------

1. Add explicit hydrogens **before** embedding. :class:`~rbfenetmap.core.models.Ligand`
   forbids implicit hydrogens, and ``AddHs(addCoords=True)`` after the fact places them
   by rule rather than by geometry -- fine for a picture, not for a starting structure.
   RDKit appends the new atoms, so donor maps built on the heavy molecule stay valid.
2. Take the correspondence from the generator. It built the molecule by editing a parent
   and therefore *knows* which atom came from where. Re-deriving it with
   ``GetSubstructMatch`` re-introduces precisely the symmetry coin-flip that
   ``match_selection="fewest_fragments"`` exists to remove. Only when a generator
   declines to supply one do we fall back to an MCS search, and the result is tagged
   ``"mcs_fallback"`` so the weaker provenance is visible in a report.
3. Seed a ``coordMap`` from the donors' *heavy* coordinates and embed with
   ``useRandomCoords``. Heavy only, because a coordMap naming every atom leaves the
   distance-bounds smoothing nothing to solve and RDKit aborts on the degenerate system;
   the mapped hydrogens are restrained in the next step instead.
4. **Restore exactness with a force field.** ``coordMap`` applies distance-bounds
   constraints, not exact placements; an embedding that satisfies every bound can still
   sit an angstrom from where it was asked to. Fixed extra points plus distance
   constraints, then a minimisation, is what actually pins the mapped atoms.
5. Rigid Kabsch fit onto the donor coordinates over the mapped set. Minimisation moves
   everything a little, and the fit removes the net drift without distorting the relaxed
   geometry the minimiser just produced.
6. Gate on the in-place ``core_rmsd`` -- the same measurement, with the same
   ``superpose_first=False``, that the pipeline will judge the resulting edges by.

Why the gate is not the safety net
----------------------------------

**You do not need to trust this module.** The in-place core RMSD check on the ``A~M`` and
``M~B`` sub-edges is already a complete test of whether ``M`` is posed in the parents'
frame: a badly posed intermediate comes back as an ordinary
:attr:`~rbfenetmap.core.models.RejectionReason.CORE_GEOMETRY_MISMATCH` and the proposal
is dropped for failing to close the gap it was invented for. The job here is to make that
check pass *often*, not to make it unnecessary.

Failures are data
-----------------

Every way this can fail returns a :class:`PoseResult` carrying a :class:`PoseRejection`,
never an exception. One molecule that will not embed among hundreds must not stop a run,
for the same reason one impossible pair does not stop the mapping stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from rbfenetmap.core.kabsch import apply_transform, core_rmsd, rigid_transform

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit import Chem

    from rbfenetmap.core.options import MappingOptions

__all__ = ("POSE_RMSD_FACTOR", "PoseDonor", "PoseRejection", "PoseResult", "pose_intermediate")

#: Fraction of ``SoftcorePolicy.core_rmsd_threshold`` an accepted pose must stay under.
#:
#: Named rather than inlined because it encodes a judgement, not a tolerance: an
#: intermediate posed at exactly the threshold would be accepted here and then rejected by
#: the very next stage, having spent an embedding and a minimisation to learn it. Half the
#: budget leaves room for the mapping the pipeline will find to differ slightly from the
#: one the generator handed over.
POSE_RMSD_FACTOR = 0.5

#: Force-constant of the distance restraints that pin mapped atoms to donor coordinates.
#: Stiff enough that the minimiser treats them as effectively rigid, finite so a strained
#: hybrid relaxes rather than exploding.
_RESTRAINT_FORCE_CONSTANT = 200.0

_MINIMIZE_ITERATIONS = 500


class PoseRejection(str, Enum):
    """Why a molecule could not be posed in its parents' frame.

    A ``str`` enum so a value drops straight into
    :attr:`~rbfenetmap.core.models.IntermediateRecord.rejection`, which is a plain string
    by design -- :class:`~rbfenetmap.core.models.RejectionReason` is the vocabulary of
    *edge* feasibility and is deliberately not reused here.

    Attributes
    ----------
    INVALID_MOLECULE
        The proposed molecule does not sanitize. A generator bug, recorded rather than
        raised so the rest of the run continues.
    CHARGE_MISMATCH
        The intermediate's net formal charge differs from a parent's. Refused outright:
        an intermediate exists to split one hard edge into two easier ones, and a charge
        change would instead split it into two harder ones.
    STEREO_UNDEFINED
        An unassigned stereocentre or double-bond configuration. Refused because nobody
        downstream can parameterise it -- the ambiguity would be resolved arbitrarily by
        whichever tool touched the molecule first.
    NO_DONOR_ATOMS
        No correspondence to any parent could be established, so there is nothing to pose
        against.
    EMBED_FAILED
        ``EmbedMolecule`` could not satisfy the distance bounds within the attempt budget.
    FORCEFIELD_FAILED
        Neither MMFF nor UFF could be set up for the molecule.
    POSE_RMSD_EXCEEDED
        A conformer was produced, but it sits too far from the donor coordinates to be
        worth handing to the feasibility stage.
    """

    INVALID_MOLECULE = "invalid_molecule"
    CHARGE_MISMATCH = "charge_mismatch"
    STEREO_UNDEFINED = "stereo_undefined"
    NO_DONOR_ATOMS = "no_donor_atoms"
    EMBED_FAILED = "embed_failed"
    FORCEFIELD_FAILED = "forcefield_failed"
    POSE_RMSD_EXCEEDED = "pose_rmsd_exceeded"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class PoseDonor:
    """One parent, and which of its atoms lend their coordinates.

    Parameters
    ----------
    name : str
        The parent ligand's name, recorded in the trace and the provenance.
    mol : rdkit.Chem.Mol
        The parent molecule, already posed in the binding-site frame.
    atom_map : Mapping[int, int], optional
        ``{intermediate atom index: parent atom index}``. ``None`` asks
        :func:`pose_intermediate` to recover a correspondence by MCS instead, which is
        strictly weaker and is reported as such.

    Notes
    -----
    Donors are consumed in order and the first to claim an intermediate atom keeps it. An
    intermediate is a hybrid, so its two parents will generally both map some of the same
    scaffold atoms; taking the earlier donor's coordinates rather than averaging is
    deliberate, since the average of two poses is a pose neither parent has.
    """

    name: str
    mol: "Chem.Mol"
    atom_map: Mapping[int, int] | None = None


@dataclass(frozen=True)
class PoseResult:
    """The outcome of one posing attempt.

    Parameters
    ----------
    mol : rdkit.Chem.Mol, optional
        The posed molecule, with explicit hydrogens and exactly one 3D conformer, or
        ``None`` when *rejection* is set.
    rmsd : float
        In-place RMSD of the mapped atoms against their donor coordinates. Reported even
        on a :attr:`PoseRejection.POSE_RMSD_EXCEEDED` rejection, because how badly the
        pose missed is the diagnostic.
    method : str
        ``"parent_atom_map"`` or ``"mcs_fallback"``.
    rejection : str, optional
        A :class:`PoseRejection` value, or ``None`` on success.
    attempts : int
        Embedding attempts spent.
    trace : tuple[str, ...]
        Human-readable log, carried into the
        :class:`~rbfenetmap.core.models.IntermediateRecord`.
    detail : Mapping[str, Any]
        Structured extras, notably ``n_mapped`` and the donors consulted.
    """

    mol: "Chem.Mol | None" = None
    rmsd: float = float("inf")
    method: str = "parent_atom_map"
    rejection: str | None = None
    attempts: int = 0
    trace: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def posed(self) -> bool:
        """Whether a usable conformer was produced."""
        return self.mol is not None


def _fail(rejection: PoseRejection, trace: Sequence[str], **extra: Any) -> PoseResult:
    """Return a rejected :class:`PoseResult`, appending the reason to *trace*."""
    return PoseResult(
        rejection=rejection.value, trace=(*trace, f"rejected: {rejection.value}"), detail=MappingProxyType(dict(extra))
    )


def _stereo_is_undefined(mol: "Chem.Mol") -> bool:
    """Whether any stereocentre or double bond is left unassigned."""
    from rdkit import Chem

    unassigned = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    if any(tag == "?" for _, tag in unassigned):
        return True
    return any(bond.GetStereo() == Chem.BondStereo.STEREOANY for bond in mol.GetBonds())


def _conformer_coords(mol: "Chem.Mol") -> np.ndarray:
    """Return the molecule's single conformer as an ``(n, 3)`` array."""
    return np.asarray(mol.GetConformer().GetPositions(), dtype=float)


def _mcs_atom_map(
    mol: "Chem.Mol", donor: "Chem.Mol", options: "MappingOptions", assigned: Mapping[int, np.ndarray]
) -> dict[int, int]:
    """Recover an intermediate-to-donor correspondence by maximum common substructure.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The intermediate, which has no conformer yet.
    donor : rdkit.Chem.Mol
        A posed parent.
    options : MappingOptions
        Supplies the ``FindMCS`` settings and the embedding cap.
    assigned : Mapping[int, numpy.ndarray]
        Coordinates already claimed by earlier donors, keyed by intermediate atom index.

    Returns
    -------
    dict[int, int]
        Possibly empty, when the two molecules share no substructure.

    Notes
    -----
    ``match_elements=True``, unlike the mappers. A mapper can afford a permissive atom
    compare because core pruning and the geometry gate sit downstream of it; here the
    correspondence *is* the pose, and a methoxy oxygen paired with a methyl carbon puts
    the whole molecule somewhere plausible-looking and wrong.

    Choosing among embeddings is where the symmetry coin-flip lives. When an earlier
    donor has already placed some of the same atoms, the candidate whose coordinates agree
    with those placements is the right one and picking it costs an RMSD each; when nothing
    has been placed yet every embedding assigns an equally defensible pose and the first
    is taken. This is why a complete ``parent_atom_map`` from the generator is worth
    having: it removes the choice rather than resolving it.
    """
    from rbfenetmap.core.mcs import mcs_embeddings, mcs_query

    pattern = mcs_query(mol, donor, options, match_elements=True)
    if pattern is None:
        return {}
    here, there = mcs_embeddings(mol, donor, pattern, options)
    if not here or not there:
        return {}

    donor_coords = _conformer_coords(donor)
    best: dict[int, int] = {}
    best_rmsd = float("inf")
    for match_here in here:
        for match_there in there:
            candidate = dict(zip(match_here, match_there))
            overlap = [idx for idx in candidate if idx in assigned]
            if not overlap:
                return candidate
            mobile = np.array([donor_coords[candidate[idx]] for idx in overlap], dtype=float)
            reference = np.array([assigned[idx] for idx in overlap], dtype=float)
            rmsd = core_rmsd(mobile, reference)
            if rmsd < best_rmsd:
                best, best_rmsd = candidate, rmsd
    return best


def pose_intermediate(
    mol: "Chem.Mol",
    donors: Sequence[PoseDonor],
    *,
    core_rmsd_threshold: float = 2.0,
    rmsd_factor: float = POSE_RMSD_FACTOR,
    seed: int = 0xF00D,
    max_attempts: int = 10,
    mapping_options: "MappingOptions | None" = None,
) -> PoseResult:
    """Give *mol* a conformer sitting in the frame its *donors* occupy.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The proposed molecule. Carries no conformer; one is added here.
    donors : Sequence[PoseDonor]
        The posed parents whose coordinates the pose is built from, in priority order.
    core_rmsd_threshold : float, optional
        The pipeline's ``SoftcorePolicy.core_rmsd_threshold``. The accept/reject bar is
        this times *rmsd_factor*.
    rmsd_factor : float, optional
        Fraction of the threshold an accepted pose must stay under. Default
        :data:`POSE_RMSD_FACTOR`.
    seed : int, optional
        Base RDKit random seed. Attempt *k* uses ``seed + k``, so retries explore rather
        than repeat.
    max_attempts : int, optional
        Embedding attempts before giving up.
    mapping_options : MappingOptions, optional
        Only consulted for the MCS fallback. Defaults are used when omitted.

    Returns
    -------
    PoseResult
        With :attr:`PoseResult.mol` set on success, or a :class:`PoseRejection` value in
        :attr:`PoseResult.rejection` on any failure. Nothing here raises.

    Examples
    --------
    >>> result = pose_intermediate(mol, [PoseDonor("lig_a", parent.mol, atom_map)])  # doctest: +SKIP
    >>> result.posed, round(result.rmsd, 3)  # doctest: +SKIP
    (True, 0.041)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    from rbfenetmap.core.options import MappingOptions

    mapping_options = mapping_options or MappingOptions()
    trace: list[str] = []

    working = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(working)
    except Exception as error:  # noqa: BLE001 - any RDKit failure here means the same thing
        return _fail(PoseRejection.INVALID_MOLECULE, trace, error=str(error))

    charge = Chem.GetFormalCharge(working)
    for donor in donors:
        donor_charge = Chem.GetFormalCharge(donor.mol)
        if donor_charge != charge:
            return _fail(
                PoseRejection.CHARGE_MISMATCH, trace, charge=charge, parent=donor.name, parent_charge=donor_charge
            )

    if _stereo_is_undefined(working):
        return _fail(PoseRejection.STEREO_UNDEFINED, trace)

    # Before embedding, never after: hydrogens placed by rule are a drawing, not a
    # geometry, and every index below has to stay valid for the donor maps.
    n_before_hydrogens = working.GetNumAtoms()
    working = Chem.AddHs(working)
    trace.append(f"added {working.GetNumAtoms() - n_before_hydrogens} explicit hydrogen(s) before embedding")

    assigned: dict[int, np.ndarray] = {}
    provenance: dict[int, str] = {}
    method = "parent_atom_map"
    for donor in donors:
        donor_coords = _conformer_coords(donor.mol)
        atom_map = donor.atom_map
        if atom_map is None:
            method = "mcs_fallback"
            atom_map = _mcs_atom_map(working, donor.mol, mapping_options, assigned)
            trace.append(f"{donor.name}: recovered {len(atom_map)} correspondence(s) by MCS")
        claimed = 0
        for here, there in atom_map.items():
            if here in assigned or here >= working.GetNumAtoms() or there >= len(donor_coords):
                continue
            assigned[here] = donor_coords[there]
            provenance[here] = donor.name
            claimed += 1
        trace.append(f"{donor.name}: donated coordinates for {claimed} atom(s)")

    if not assigned:
        return _fail(PoseRejection.NO_DONOR_ATOMS, trace, donors=[d.name for d in donors])

    # Heavy atoms only, and only here. A coordMap naming *every* atom of a molecule
    # leaves the distance-bounds smoothing nothing to solve for, and RDKit's optimiser
    # aborts on the degenerate system rather than returning a failure code. The mapped
    # hydrogens are not discarded -- they are restrained below and they count towards the
    # RMSD gate; they simply do not need to constrain an embedding their heavy parents
    # already determine.
    embed_map = {idx: xyz for idx, xyz in assigned.items() if working.GetAtomWithIdx(idx).GetAtomicNum() != 1}
    if not embed_map:
        embed_map = dict(assigned)
    coord_map = {idx: Point3D(*(float(v) for v in xyz)) for idx, xyz in embed_map.items()}

    status, attempts = -1, 0
    for attempt in range(max(1, max_attempts)):
        attempts = attempt + 1
        try:
            status = AllChem.EmbedMolecule(
                working, coordMap=coord_map, randomSeed=seed + attempt, useRandomCoords=True, clearConfs=True
            )
        except Exception:  # noqa: BLE001 - ETKDG raises on some degenerate systems; that is a failure like any other
            status = -1
        if status >= 0:
            break
    if status < 0 or working.GetNumConformers() == 0:
        return _fail(PoseRejection.EMBED_FAILED, trace, attempts=attempts)
    trace.append(f"embedded on attempt {attempts} of {max(1, max_attempts)}")

    # coordMap is a set of distance bounds, not a set of placements. Without this step an
    # embedding that satisfies every bound can still sit an angstrom off, and the rigid
    # fit below would then move the whole molecule to split the difference.
    field_ = None
    try:
        properties = AllChem.MMFFGetMoleculeProperties(working)
        if properties is not None:
            field_ = AllChem.MMFFGetMoleculeForceField(working, properties)
        if field_ is None:
            field_ = AllChem.UFFGetMoleculeForceField(working)
    except Exception:  # noqa: BLE001 - an unparameterisable hybrid is data, not an error
        field_ = None
    if field_ is None:
        return _fail(PoseRejection.FORCEFIELD_FAILED, trace)

    for idx, xyz in assigned.items():
        point = field_.AddExtraPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]), fixed=True) - 1
        field_.AddDistanceConstraint(point, idx, 0.0, 0.0, _RESTRAINT_FORCE_CONSTANT)
    field_.Initialize()
    field_.Minimize(maxIts=_MINIMIZE_ITERATIONS)
    trace.append(f"restrained {len(assigned)} atom(s) and minimised")

    indices = sorted(assigned)
    coords = _conformer_coords(working)
    reference = np.array([assigned[idx] for idx in indices], dtype=float)
    rotation, translation = rigid_transform(coords[indices], reference)
    coords = apply_transform(coords, rotation, translation)

    conformer = working.GetConformer()
    for idx, xyz in enumerate(coords):
        conformer.SetAtomPosition(idx, Point3D(*(float(v) for v in xyz)))

    rmsd = core_rmsd(coords[indices], reference)
    trace.append(f"in-place RMSD over {len(indices)} mapped atom(s): {rmsd:.3f} A")

    detail = MappingProxyType(
        {
            "n_mapped": len(indices),
            "donors": [d.name for d in donors],
            "atom_donor": {str(k): v for k, v in sorted(provenance.items())},
        }
    )
    limit = core_rmsd_threshold * rmsd_factor
    if rmsd > limit:
        trace.append(f"rejected: {PoseRejection.POSE_RMSD_EXCEEDED.value} ({rmsd:.3f} > {limit:.3f} A)")
        return PoseResult(
            rmsd=rmsd,
            method=method,
            rejection=PoseRejection.POSE_RMSD_EXCEEDED.value,
            attempts=attempts,
            trace=tuple(trace),
            detail=detail,
        )
    return PoseResult(mol=working, rmsd=rmsd, method=method, attempts=attempts, trace=tuple(trace), detail=detail)
