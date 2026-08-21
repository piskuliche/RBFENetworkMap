"""What an intermediate generator proposes, and how a proposal becomes a ligand.

A generator invents a molecule that sits *between* two ligands the mapper cannot relate
cheaply, turning one hard edge into two easier ones. This module holds the vocabulary it
speaks in -- :class:`ProposedMolecule`, :class:`ProposedLink`,
:class:`IntermediateProposal` -- plus the naming rule and the one function that turns a
proposal into a real :class:`~rbfenetmap.core.models.Ligand`.

These types live here rather than in :mod:`rbfenetmap.core.models` on purpose: they carry
RDKit molecules and an atom-level correspondence, and ``models`` is deliberately free of
chemistry so the data model stays importable and testable without one.

Three constraints the types enforce, and why
--------------------------------------------

**A proposed molecule carries no conformer.** Posing is centralised in
:mod:`rbfenetmap.core.posing`, for the same reason :mod:`rbfenetmap.core.descriptors`
centralises scoring inputs: a generator that poses its own output makes the quality of
every intermediate depend on which generator produced it, and makes the pose
unauditable. A generator that *knows* where its atoms belong says so through a complete
:attr:`ProposedMolecule.parent_atom_map`, which is strictly more useful than coordinates
-- the poser can act on a correspondence, and can tell you which parent every atom came
from afterwards.

**A link's hint is advisory and can never become an**
:attr:`~rbfenetmap.core.models.EdgeScore.total`. A generator may know that one of its
proposals is more promising than another, and that is worth recording; it is not a cost.
Letting a hint reach a score would put a second scoring system in the package on a
different scale, which is the dual of the rule that a scorer must not invent rejections.

**A proposal's rejection is a plain string.**
:class:`~rbfenetmap.core.models.RejectionReason` is the vocabulary of *edge* feasibility.
Reusing it here would make ``core_geometry_mismatch`` mean two different things depending
on which object it was read from.

Naming
------

:func:`intermediate_name` is content-addressed -- ``int_{a}_{b}_{sha1(smiles)[:6]}`` with
the parents sorted -- rather than counter-based. The same intermediate is often reachable
from the same gap by two routes, and a hash makes that a natural dedupe instead of a pair
of near-identical ligands; it also keeps a run reproducible under ``jobs > 1``, where a
counter's value depends on which worker finished first.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from rbfenetmap.core.models import Ligand, LigandProvenance
from rbfenetmap.core.posing import POSE_RMSD_FACTOR, PoseDonor, PoseResult, pose_intermediate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit import Chem

    from rbfenetmap.core.options import MappingOptions, SoftcorePolicy

__all__ = (
    "INTERMEDIATE_KIND",
    "INTERMEDIATE_NAME_PREFIX",
    "IntermediateOptions",
    "IntermediateProposal",
    "ProposedLink",
    "ProposedMolecule",
    "intermediate_name",
    "reserve_intermediate_names",
    "synthesize_ligand",
)

#: The :attr:`~rbfenetmap.core.models.LigandProvenance.kind` this package writes.
INTERMEDIATE_KIND = "intermediate"

#: Prefix every generated ligand name carries. Matches ``Ligand``'s name pattern and
#: contains no :data:`~rbfenetmap.core.models.EDGE_SEPARATOR`, so a generated name
#: survives :func:`~rbfenetmap.core.models.edge_key` and
#: :func:`~rbfenetmap.core.models.parse_edge_key` unchanged.
INTERMEDIATE_NAME_PREFIX = "int_"

_HASH_LENGTH = 6


@dataclass(frozen=True)
class IntermediateOptions:
    """Settings a generator and the poser share.

    Parameters
    ----------
    max_molecules : int, optional
        Cap on how many molecules a generator may propose for one gap. A generator that
        enumerates every single-substituent swap on a heavily decorated pair can produce
        dozens; each one costs an embedding and a minimisation, so the cap is the knob
        that keeps generation from dominating a run.
    seed : int, optional
        Base RDKit random seed for posing, so a run is reproducible.
    max_pose_attempts : int, optional
        Embedding attempts spent on one molecule before giving up on it.
    pose_rmsd_factor : float, optional
        Fraction of ``SoftcorePolicy.core_rmsd_threshold`` an accepted pose must stay
        under. See :data:`~rbfenetmap.core.posing.POSE_RMSD_FACTOR`.

    Raises
    ------
    ValueError
        If any value is out of range.

    Notes
    -----
    Deliberately *not* on :class:`~rbfenetmap.core.options.NetworkOptions`. Nothing in
    this phase runs generation from the pipeline, and adding a field to the serialized
    options block would change the bytes of every network file written -- including the
    all-real ones that must stay byte-identical. The pipeline-facing knobs land with the
    pipeline that reads them.
    """

    max_molecules: int = 4
    seed: int = 0xF00D
    max_pose_attempts: int = 10
    pose_rmsd_factor: float = POSE_RMSD_FACTOR

    def __post_init__(self) -> None:
        """Reject nonsensical budgets up front."""
        if self.max_molecules < 1:
            raise ValueError("max_molecules must be at least 1.")
        if self.max_pose_attempts < 1:
            raise ValueError("max_pose_attempts must be at least 1.")
        if not 0.0 < self.pose_rmsd_factor <= 1.0:
            raise ValueError("pose_rmsd_factor must lie in (0, 1].")


@dataclass(frozen=True)
class ProposedMolecule:
    """One molecule a generator suggests inserting into a gap.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule, **without a conformer**. Any conformers present are stripped at
        construction rather than rejected, so a generator that happened to build its
        molecule from a posed parent does not have to remember to clear them -- but the
        coordinates are discarded either way, because posing is centralised.
    parents : tuple[str, ...]
        Names of the ligands it was derived from, sorted at construction.
    parent_atom_map : Mapping[str, Mapping[int, int]]
        ``{parent name: {proposed atom index: parent atom index}}``. May be empty, or
        cover only some parents; the poser falls back to an MCS search for whatever is
        missing and records that it had to.
    hint : float, optional
        The generator's own ordering preference among its proposals, lower being more
        promising. Advisory only -- see the module docstring.
    detail : Mapping[str, Any], optional
        Free-form annotations carried into the ligand's provenance.

    Raises
    ------
    ValueError
        If *parents* is empty, or *parent_atom_map* names a parent that is not in
        *parents*.
    """

    mol: "Chem.Mol"
    parents: tuple[str, ...]
    parent_atom_map: Mapping[str, Mapping[int, int]] = field(default_factory=lambda: MappingProxyType({}))
    hint: float | None = None
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Sort the parents, strip conformers, and check the map's keys."""
        object.__setattr__(self, "parents", tuple(sorted(self.parents)))
        if not self.parents:
            raise ValueError("A ProposedMolecule must name at least one parent ligand.")
        unknown = sorted(set(self.parent_atom_map) - set(self.parents))
        if unknown:
            raise ValueError(f"parent_atom_map names non-parent ligand(s) {unknown}; parents are {list(self.parents)}.")
        if self.mol is not None and self.mol.GetNumConformers():
            stripped = type(self.mol)(self.mol)
            stripped.RemoveAllConformers()
            object.__setattr__(self, "mol", stripped)


@dataclass(frozen=True)
class ProposedLink:
    """A sub-edge the generator expects its molecule to make possible.

    Parameters
    ----------
    source, target : str
        Endpoints. One is normally the proposed molecule, the other a parent.
    hint : float, optional
        How promising the generator believes the link to be, lower being better.
    detail : Mapping[str, Any], optional
        Free-form annotations.

    Notes
    -----
    **The hint can never become an**
    :attr:`~rbfenetmap.core.models.EdgeScore.total`. Nothing in this package reads it as a
    cost, and nothing should: a generator's opinion about its own output is on a scale
    only that generator knows, so promoting it to a score would mean two incomparable
    numbers competing inside the planner's edge ordering. The link exists to say *which
    edges are worth evaluating*; the scorer says what they cost.
    """

    source: str
    target: str
    hint: float | None = None
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class IntermediateProposal:
    """A generator's complete answer for one gap.

    Parameters
    ----------
    source, target : str
        The gap the generator was asked about.
    generator : str
        Registered plugin name.
    molecules : tuple[ProposedMolecule, ...], optional
        What it suggests inserting. Empty is a legitimate answer.
    links : tuple[ProposedLink, ...], optional
        Sub-edges it expects to become feasible.
    rejection : str, optional
        Why it proposed nothing. A plain ``str``, never a
        :class:`~rbfenetmap.core.models.RejectionReason`.
    trace : tuple[str, ...], optional
        Human-readable log of what it tried.

    Raises
    ------
    ValueError
        If the two endpoints are the same ligand.
    """

    source: str
    target: str
    generator: str
    molecules: tuple[ProposedMolecule, ...] = ()
    links: tuple[ProposedLink, ...] = ()
    rejection: str | None = None
    trace: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a proposal for a gap between a ligand and itself."""
        if self.source == self.target:
            raise ValueError(f"An intermediate proposal needs two distinct ligands; got {self.source!r} twice.")

    @property
    def proposed(self) -> bool:
        """Whether the generator suggested anything at all."""
        return bool(self.molecules)


def intermediate_name(parents: Sequence[str], mol: "Chem.Mol") -> str:
    """Return the content-addressed name for an invented molecule.

    Parameters
    ----------
    parents : Sequence[str]
        Names of the ligands it bridges. Sorted here, so the caller need not.
    mol : rdkit.Chem.Mol
        The molecule. Only its canonical SMILES is used, so a name is stable across a
        re-embedding and across atom reordering.

    Returns
    -------
    str
        ``int_{a}_{b}_{sha1(canonical_smiles)[:6]}``.

    Raises
    ------
    ValueError
        If *parents* is empty, or the resulting name would not be a legal ligand name.

    Notes
    -----
    Content-addressed rather than counter-based, and the parents are part of the address
    rather than only the structure. The hash makes the same molecule proposed twice for
    the same gap collapse to one ligand -- which is the common case, since a gap is
    usually reachable from either end. The parent tokens stay in because two gaps that
    happen to want the same molecule want it for different reasons, and a name that hid
    that would leave two different provenances competing for one vertex.

    Hydrogens are suppressed before canonicalisation. A generator that hands over a
    molecule with explicit hydrogens and one that leaves them implicit are proposing the
    same thing, and they must not get different names for it.
    """
    from rdkit import Chem

    from rbfenetmap.core.models import _LIGAND_NAME_RE

    if not parents:
        raise ValueError("intermediate_name needs at least one parent name.")
    try:
        bare = Chem.RemoveHs(Chem.Mol(mol))
    except Exception:  # noqa: BLE001 - a molecule too broken to strip is named from itself
        bare = mol
    digest = hashlib.sha1(Chem.MolToSmiles(bare).encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    name = INTERMEDIATE_NAME_PREFIX + "_".join((*sorted(parents), digest))
    if not _LIGAND_NAME_RE.match(name):
        raise ValueError(
            f"Generated intermediate name {name!r} is not a legal ligand name. Parent names are "
            "restricted to the same character class, so this means one of them is not."
        )
    return name


def reserve_intermediate_names(names: Iterable[str], *, enabled: bool) -> None:
    """Refuse user ligand names that would collide with generated ones.

    Parameters
    ----------
    names : Iterable[str]
        The user's ligand names.
    enabled : bool
        Whether intermediate generation is switched on for this run.

    Raises
    ------
    ValueError
        If *enabled* and any name starts with :data:`INTERMEDIATE_NAME_PREFIX`.

    Notes
    -----
    The prefix is reserved **only when the feature is on**, which is the whole point of
    taking *enabled* rather than reading it from a global. A user with a ligand honestly
    called ``int_3`` is running a plain network plan that cannot possibly generate a
    conflicting name, and refusing it would be a compatibility break bought for nothing.
    """
    if not enabled:
        return
    clashes = sorted(name for name in names if name.startswith(INTERMEDIATE_NAME_PREFIX))
    if clashes:
        raise ValueError(
            f"Ligand name(s) {clashes} start with the reserved prefix {INTERMEDIATE_NAME_PREFIX!r}, which "
            "intermediate generation uses for the molecules it invents. Rename them, or plan without "
            "intermediate generation, where the prefix is not reserved."
        )


def synthesize_ligand(
    proposed: ProposedMolecule,
    parents: Mapping[str, Ligand],
    *,
    generator: str,
    softcore: "SoftcorePolicy | None" = None,
    options: IntermediateOptions | None = None,
    mapping_options: "MappingOptions | None" = None,
) -> tuple[Ligand | None, PoseResult]:
    """Pose *proposed* against its parents and wrap the result in a ligand.

    Parameters
    ----------
    proposed : ProposedMolecule
        The conformer-free molecule a generator suggested.
    parents : Mapping[str, Ligand]
        The real ligands, keyed by name. Must contain every name in
        :attr:`ProposedMolecule.parents`.
    generator : str
        Registered name of the generator, recorded in the provenance.
    softcore : SoftcorePolicy, optional
        Supplies ``core_rmsd_threshold``. Defaults are used when omitted.
    options : IntermediateOptions, optional
        Supplies the seed, the attempt budget, and the RMSD factor.
    mapping_options : MappingOptions, optional
        Only consulted by the poser's MCS fallback.

    Returns
    -------
    ligand : Ligand or None
        ``None`` when the pose was rejected; the :class:`PoseResult` says why.
    result : PoseResult
        Always returned, successful or not, because the trace is what a user needs when
        an intermediate does not appear.

    Raises
    ------
    KeyError
        If *parents* does not contain a name the proposal claims. That is a caller bug
        rather than a chemistry outcome, so it raises where posing failures do not.

    Notes
    -----
    The parents are handed to the poser in sorted-name order, which is also the order
    :func:`intermediate_name` uses. That matters because donors are consumed in priority
    order: an unordered iteration would let two runs of the same input take scaffold
    coordinates from different parents and produce two slightly different poses under the
    same name.
    """
    from rbfenetmap.core.options import SoftcorePolicy

    options = options or IntermediateOptions()
    softcore = softcore or SoftcorePolicy()

    donors = [
        PoseDonor(name=name, mol=parents[name].mol, atom_map=proposed.parent_atom_map.get(name))
        for name in proposed.parents
    ]
    result = pose_intermediate(
        proposed.mol,
        donors,
        core_rmsd_threshold=softcore.core_rmsd_threshold,
        rmsd_factor=options.pose_rmsd_factor,
        seed=options.seed,
        max_attempts=options.max_pose_attempts,
        mapping_options=mapping_options,
    )
    if result.mol is None:
        return None, result

    provenance = LigandProvenance(
        kind=INTERMEDIATE_KIND,
        generator=generator,
        parents=proposed.parents,
        pose_method=result.method,
        pose_rmsd=result.rmsd,
        detail=MappingProxyType({**dict(proposed.detail), **dict(result.detail)}),
    )
    ligand = Ligand.synthesized(result.mol, intermediate_name(proposed.parents, result.mol), provenance)
    return ligand, result
